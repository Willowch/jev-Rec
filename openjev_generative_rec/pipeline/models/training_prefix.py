"""Differentiable prefix reuse for single-GPU Qwen3.5 SID SFT."""
import os
from contextlib import contextmanager
from copy import copy
from itertools import chain

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

from .openjev_runtime import backward_choice, choice_loss, measure_phase, prepare_candidate_chunks
from jev.prefix_cache import common_prefix_length


class TrainingLinearState(LinearAttentionLayer):
    def update_conv_state(self, conv_states, **kwargs):
        # Qwen supplies the complete convolution window. Keep the tensor's graph and avoid in-place writes.
        self.conv_states = conv_states
        self.is_conv_states_initialized = True
        self.has_previous_state = True
        return conv_states

    def update_recurrent_state(self, recurrent_states, **kwargs):
        self.recurrent_states = recurrent_states
        self.is_recurrent_states_initialized = True
        return recurrent_states


@contextmanager
def suspend_checkpointing(backbone):
    layers = [layer for layer in backbone.modules() if getattr(layer, "gradient_checkpointing", False)]
    for layer in layers:
        layer.gradient_checkpointing = False
    try:
        yield
    finally:
        for layer in layers:
            layer.gradient_checkpointing = True


def training_cache(config):
    cache = DynamicCache(config=config)
    cache.layers = [TrainingLinearState(config) if type(layer) is LinearAttentionLayer else layer
                    for layer in cache.layers]
    return cache


def cache_tensors(cache, *, require_grad=True):
    states = {}
    for index, layer in enumerate(cache.layers):
        if isinstance(layer, TrainingLinearState):
            names = ("conv_states", "recurrent_states")
        elif type(layer) is DynamicLayer:
            names = ("keys", "values")
        else:
            raise ValueError(f"Unsupported Qwen training cache layer {type(layer).__name__}")
        for name in names:
            state = getattr(layer, name)
            if state is None or state.shape[0] != 1 or (require_grad and not state.requires_grad):
                raise RuntimeError(f"Shared-prefix state {index}.{name} is missing or cannot receive gradients")
            states[index, name] = state
    return states


def suffix_logits(model, prefix_cache, states, input_ids, attention_mask, last_token):
    branch = copy(prefix_cache)
    branch.layers = [copy(layer) for layer in prefix_cache.layers]
    for (index, name), tensor in states.items():
        setattr(branch.layers[index], name, tensor.repeat_interleave(input_ids.shape[0], dim=0))
    output = model.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            past_key_values=branch, use_cache=True, return_dict=True)
    indices = torch.arange(len(last_token), device=input_ids.device)
    hidden = output.last_hidden_state[indices, last_token]
    return model.head(hidden.float()).squeeze(-1).float()


def token_sequences(prepared):
    return [tokens[:length].tolist() for _, (encoded, _, _) in prepared
            for tokens, length in zip(encoded["input_ids"], encoded["attention_mask"].sum(-1).tolist())]


def prefix_cut(sequences):
    # Keep at least two suffix tokens and align with the gated-delta chunk boundary.
    cut = min(common_prefix_length(sequences), min(map(len, sequences)) - 2) // 64 * 64
    return cut if cut >= 64 and cut * 2 >= max(map(len, sequences)) else 0


def forward_token_count(sequences, batch_size, cut):
    """Estimate padded candidate tokens, including the scoring pass for multi-batch losses."""
    count = 0
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        count += len(batch) * (max(map(len, batch)) - cut)
    return count if len(sequences) <= batch_size else 2 * count


def make_suffix_batches(sequences, cut, batch_size, pad, device):
    batches = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        width = max(map(len, batch))
        suffix = torch.tensor([seq[cut:] + [pad] * (width - len(seq)) for seq in batch], device=device)
        lengths = torch.tensor([len(seq) for seq in batch], device=device)
        mask = (torch.arange(width, device=device) < lengths[:, None]).long()
        batches.append((start, suffix, mask, lengths - cut - 1))
    return batches


def accumulate_state_grads(leaves, state_grads):
    for name, leaf in leaves.items():
        if leaf.grad is not None:
            if name not in state_grads:
                state_grads[name] = leaf.grad.to(torch.float32)
            else:
                state_grads[name].add_(leaf.grad)
            leaf.grad = None


def backward_shared_choices(model, rows, groups, cut, batch_size, brier_weight, scale, *, timings, audit, usage):
    device = model.head.weight.device
    tokens = torch.tensor([groups[0][0][:cut]], device=device)
    losses = []
    recompute_prefix = os.environ.get("OPENJEV_RECOMPUTE_PREFIX") == "1"

    with suspend_checkpointing(model.backbone):
        with torch.set_grad_enabled(not recompute_prefix), measure_phase(timings, "prefix_forward", cuda=True):
            cache = training_cache(model.backbone.config)
            result = model.backbone(input_ids=tokens, past_key_values=cache, use_cache=True, return_dict=True)
            saved = result.past_key_values
            del result
            states = cache_tensors(saved, require_grad=not recompute_prefix)
            leaves = {name: value.detach().requires_grad_() for name, value in states.items()}

        state_grads = {}
        for row, sequences in zip(rows, groups):
            batches = make_suffix_batches(sequences, cut, batch_size, model.tokenizer.pad_token_id, device)
            if len(batches) == 1:
                _, ids, mask, last = batches[0]
                with measure_phase(timings, "gradient_forward", cuda=True):
                    logits = suffix_logits(model, saved, leaves, ids, mask, last)
                loss = choice_loss(logits, row, brier_weight)
                with measure_phase(timings, "backward", cuda=True):
                    (loss * scale).backward()
                    accumulate_state_grads(leaves, state_grads)
            else:
                with torch.no_grad(), measure_phase(timings, "scoring_forward", cuda=True):
                    logits = torch.cat([suffix_logits(model, saved, leaves, ids, mask, last)
                                        for _, ids, mask, last in batches])
                logits.requires_grad_()
                loss = choice_loss(logits, row, brier_weight)
                gradient, = torch.autograd.grad(loss * scale, logits)
                for start, ids, mask, last in batches:
                    with measure_phase(timings, "gradient_forward", cuda=True):
                        scores = suffix_logits(model, saved, leaves, ids, mask, last)
                    with measure_phase(timings, "backward", cuda=True):
                        scores.backward(gradient[start:start + ids.shape[0]])
                        accumulate_state_grads(leaves, state_grads)
            losses.append(float(loss.detach()))
            if audit is not None:
                audit(row, logits)

        with measure_phase(timings, "prefix_backward", cuda=True):
            if not state_grads:
                raise RuntimeError("Candidate branches did not backpropagate into the shared prefix")
            if recompute_prefix:
                del saved, cache, leaves, states
                cache = training_cache(model.backbone.config)
                result = model.backbone(input_ids=tokens, past_key_values=cache, use_cache=True, return_dict=True)
                states = cache_tensors(result.past_key_values)
                del result
            torch.autograd.backward([states[name] for name in state_grads],
                                    [grad.to(states[name].dtype) for name, grad in state_grads.items()])

    if usage is not None:
        usage["choices"] += len(rows)
        usage["prefills"] = usage.get("prefills", 0) + 1
        usage["forward_tokens_saved"] += (sum(map(len, groups)) - 1) * cut
        counts = usage.setdefault("choices_by_layer", {})
        for row in rows:
            level = str(row["state"]["semantic_id_layer"])
            counts[level] = counts.get(level, 0) + 1
    return sum(losses)


def first_layer_cut(row, sequences, chunk_size):
    if row["state"]["semantic_id_layer"] == 1 and len(sequences) > chunk_size:
        return prefix_cut(sequences)
    return 0


def backward_prefix_choice(model, row, chunk_size=1, brier_weight=0.1, scale=1.0, *,
                           prefix_batch_size=None, timings=None, audit=None, usage=None):
    if row["state"]["semantic_id_layer"] != 1 or len(row["options"]) <= chunk_size:
        return backward_choice(model, row, chunk_size, brier_weight, scale, reuse_encoding=True,
                               timings=timings, audit=audit)

    with measure_phase(timings, "candidate_encoding"):
        prepared = prepare_candidate_chunks(model, row, chunk_size)
        sequences = token_sequences(prepared)
        cut = prefix_cut(sequences)
    if not cut:
        return backward_choice(model, row, chunk_size, brier_weight, scale, prepared=prepared,
                               timings=timings, audit=audit)
    return backward_shared_choices(model, [row], [sequences], cut, prefix_batch_size or chunk_size,
                                   brier_weight, scale, timings=timings, audit=audit, usage=usage)


def backward_prefix_all(model, rows, chunk_size, brier_weight, scale, *, prefix_batch_size, timings, audit, usage):
    if not rows:
        return 0.0
    if len(rows) == 1:
        return backward_prefix_choice(model, rows[0], chunk_size, brier_weight, scale,
                                      prefix_batch_size=prefix_batch_size, timings=timings, audit=audit, usage=usage)

    with measure_phase(timings, "candidate_encoding"):
        prepared = [prepare_candidate_chunks(model, row, chunk_size) for row in rows]
        groups = [token_sequences(batches) for batches in prepared]
        cut = prefix_cut(list(chain.from_iterable(groups)))
        first_only = [first_layer_cut(row, sequences, chunk_size) for row, sequences in zip(rows, groups)]
        baseline = sum(length + forward_token_count(group, prefix_batch_size if length else chunk_size, length)
                       for group, length in zip(groups, first_only))
        shared = cut + sum(forward_token_count(group, prefix_batch_size, cut) for group in groups)
    if cut and shared < baseline:
        return backward_shared_choices(model, rows, groups, cut, prefix_batch_size, brier_weight, scale,
                                       timings=timings, audit=audit, usage=usage)

    loss = 0.0
    for row, group, batches, length in zip(rows, groups, prepared, first_only):
        if length:
            loss += backward_shared_choices(model, [row], [group], length, prefix_batch_size, brier_weight, scale,
                                             timings=timings, audit=audit, usage=usage)
        else:
            loss += backward_choice(model, row, chunk_size, brier_weight, scale, prepared=batches,
                                    timings=timings, audit=audit)
    return loss
