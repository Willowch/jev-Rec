"""Evaluate one SFT choice across GPUs while preserving its full loss and update."""
import os
from datetime import timedelta

import torch
from torch import distributed as dist

from ..models.openjev_runtime import backward_logits, choice_loss, measure_phase, prepare_candidate_chunks, score

PHASES = ("candidate_encoding", "scoring_forward", "gradient_forward", "backward",
          "logit_communication", "gradient_communication", "optimizer")


class CandidateParallel:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device

    @classmethod
    def from_environment(cls):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size < 2 or world_size != local_world_size or rank != local_rank:
            raise ValueError("Candidate parallel SFT requires single-node torchrun with at least two workers")
        if torch.cuda.device_count() < world_size or not 0 <= local_rank < world_size:
            raise ValueError("Candidate parallel SFT requires one visible CUDA GPU per worker")
        if os.environ.get("JEV_DEVICE_MAP"):
            raise ValueError("Candidate parallel SFT places one complete model on each GPU; unset JEV_DEVICE_MAP")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", str(world_size)))
        torch.set_num_threads(max(1, cpus // world_size))
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
        return cls(rank, world_size, device)

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @torch.no_grad()
    def synchronize_parameters(self, parameters):
        for parameter in parameters:
            dist.broadcast(parameter, src=0)

    def backward_choice(self, model, row, chunk_size, brier_weight, scale, *, timings=None, audit=None):
        if row["target"] == [1.0]:
            return 0.0
        count = len(row["options"])
        start = count * self.rank // self.world_size
        stop = count * (self.rank + 1) // self.world_size
        span = slice(start, stop)
        shard = {**row, "options": row["options"][span]}
        all_logits = torch.zeros(count, dtype=torch.float32, device=self.device)
        local_graph = None
        if start < stop:
            with measure_phase(timings, "candidate_encoding"):
                prepared = prepare_candidate_chunks(model, shard, chunk_size)
            if len(prepared) == 1:
                _, inputs = prepared[0]
                with measure_phase(timings, "gradient_forward", cuda=True):
                    local_graph = model.score_encoded(*inputs)[0].float()
                all_logits[span] = local_graph.detach()
            else:
                with measure_phase(timings, "scoring_forward", cuda=True):
                    all_logits[span] = score(model, shard, chunk_size, prepared=prepared)

        # Each score slot has exactly one contributing rank. Preserve catalog order.
        with measure_phase(timings, "logit_communication", cuda=True):
            dist.all_reduce(all_logits, op=dist.ReduceOp.SUM)
        all_logits.requires_grad_(True)
        loss = choice_loss(all_logits, row, brier_weight)
        gradient, = torch.autograd.grad(loss * scale, all_logits)
        if local_graph is not None:
            with measure_phase(timings, "backward", cuda=True):
                local_graph.backward(gradient[span])
        elif start < stop:
            backward_logits(model, shard, gradient[span], chunk_size, prepared=prepared, timings=timings)
        if audit is not None:
            audit(row, all_logits)
        return float(loss.detach())

    @torch.no_grad()
    def synchronize_gradients(self, parameters):
        # Use FP32 for the sum. The final slots distinguish absent gradients from zero gradients.
        sizes = [parameter.numel() for parameter in parameters]
        total = sum(sizes)
        packed = torch.zeros(total + len(parameters), device=self.device, dtype=torch.float32)
        slots = packed[:total].split(sizes)
        present = packed[total:]
        for index, (parameter, slot) in enumerate(zip(parameters, slots)):
            if parameter.grad is not None:
                slot.copy_(parameter.grad.reshape(-1))
                present[index] = 1
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        for parameter, slot, has_gradient in zip(parameters, slots, present.tolist()):
            parameter.grad = slot.view_as(parameter).to(parameter.dtype) if has_gradient else None

    def peak_memory(self):
        peaks = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                             device=self.device, dtype=torch.int64)
        dist.reduce(peaks, dst=0, op=dist.ReduceOp.MAX)
        return tuple(value / 2**30 for value in peaks.tolist()) if self.rank == 0 else (None, None)

    def max_phase_timings(self, timings):
        values = torch.tensor([timings.get(phase, 0.0) for phase in PHASES], device=self.device, dtype=torch.float64)
        dist.reduce(values, dst=0, op=dist.ReduceOp.MAX)
        return dict(zip(PHASES, values.tolist())) if self.rank == 0 else {}
