"""Run GSPO with a supervised target in each group on the SID projection model."""

import argparse
import gc
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from filelock import FileLock

from ..common import enable_comparison, sha256
from ..data_preparation.build_jev_data import SequenceData
from ..data_preparation.embedding_cache import encoder_fingerprint
from ..models.openjev_runtime import set_training_mode
from .gspo import group_advantages, gspo_loss
from .gspo_selection import checkpoint_digest
from .rewards import MiniOneRecRankedReward
from .sid_decoder import (METHOD, audit_histories, load_projection_checkpoint, log, memory, optimizer_for,
                          prepare_train, ranking_batches, require_formal_cohort, runtime_identity,
                          save_checkpoint, synchronize, test)

enable_comparison()
from comparison.protocol import (MAX_HISTORY, TRAIN_SELECTION, VALIDATION_K, load_cohort, read_json,
                                 sampled_users, staged_output, validation_metrics, validation_record, write_json)


SEED = 43
GROUP_SIZE = 16
SAMPLED = GROUP_SIZE - 1
GROUPS_PER_ROLLOUT = 4
MICROBATCH = 4
NLL_WEIGHT = 0.1
CLIP_LOW = 3e-4
CLIP_HIGH = 4e-4
LR_SCALE = 0.5


def release_device(args):
    gc.collect()
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class Group:
    prompt: list
    codes: list
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor


def score_codes(decoder, prompt, codes):
    return decoder.sequence_log_probs([prompt] * len(codes), codes)[0]


@torch.no_grad()
def collect_group(decoder, prompt, target, generator, reward):
    sampled = decoder.sample_codes(prompt, SAMPLED, MICROBATCH, generator)
    codes = [*sampled, target]
    parts = [score_codes(decoder, prompt, sampled[start:start + MICROBATCH])
             for start in range(0, SAMPLED, MICROBATCH)]
    parts.append(score_codes(decoder, prompt, [target]))
    old = torch.cat(parts).cpu()
    if old.shape != (GROUP_SIZE, decoder.depth) or not torch.isfinite(old).all():
        raise ValueError("Expected 16 finite old-policy trajectory scores")
    rewards = torch.tensor(reward.score_ranked(codes, target, old.sum(dim=1).tolist()), dtype=torch.float32)
    return Group(prompt, codes, old, rewards, group_advantages(rewards))


def train_group(decoder, group, optimizer, parameters):
    optimizer.zero_grad(set_to_none=True)
    policy_loss, ratio, clipped, squared_grad = 0.0, 0.0, 0.0, 0.0
    for start in range(0, SAMPLED, MICROBATCH):
        stop = min(start + MICROBATCH, SAMPLED)
        log_probs = score_codes(decoder, group.prompt, group.codes[start:stop])
        old = group.old_log_probs[start:stop].to(log_probs.device)
        advantage = group.advantages[start:stop].to(log_probs.device)
        loss, metrics = gspo_loss(log_probs, old, advantage, CLIP_LOW, CLIP_HIGH)
        weight = (stop - start) / SAMPLED
        weighted = loss * weight
        grad, = torch.autograd.grad(weighted, log_probs, retain_graph=True)
        squared_grad += float(grad.detach().square().sum())
        weighted.backward()
        policy_loss += float(weighted.detach())
        ratio += weight * metrics["ratio_mean"]
        clipped += weight * metrics["clip_fraction"]
    target = group.codes[-1]
    nll = decoder.training_loss([group.prompt], [target])
    (NLL_WEIGHT * nll).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": policy_loss + NLL_WEIGHT * float(nll.detach()), "gspo_loss": policy_loss,
            "supervised_nll": float(nll.detach()), "policy_logprob_grad_norm": math.sqrt(squared_grad),
            "gradient_norm": float(grad_norm), "ratio_mean": ratio, "clip_fraction": clipped,
            "natural_hits": sum(code == target for code in group.codes[:SAMPLED]),
            "reward_std": float(group.rewards.std(unbiased=False)),
            "reward_mean": float(group.rewards.mean()), "forced_reward": float(group.rewards[-1])}


def train(args, data, identity, source, checkpoint, train_users, valid_users):
    decoder = load_projection_checkpoint(args, data, checkpoint, source["inputs"], valid_users, train=True)
    optimizer = optimizer_for(decoder.decision, decoder.heads.parameters())
    for group in optimizer.param_groups:
        group["lr"] *= LR_SCALE
    parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    generator = torch.Generator(device=decoder.device).manual_seed(SEED)
    reward = MiniOneRecRankedReward()
    samples = prepare_train(decoder, train_users)
    anchors = audit_histories(valid_users)
    steps = (len(samples) // 2, len(samples))
    paths = {}
    natural_hits = hit_groups = nonzero_grad = 0
    synchronize(decoder.device)
    started = perf_counter()
    if decoder.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(decoder.device)
    with (args.output / "training.jsonl").open("x", encoding="utf-8") as stream:
        for start in range(0, len(samples), GROUPS_PER_ROLLOUT):
            count = min(GROUPS_PER_ROLLOUT, len(samples) - start)
            groups = [collect_group(decoder, prompt, target, generator, reward)
                      for prompt, target in samples[start:start + count]]
            for step, group in enumerate(groups, start + 1):
                record = train_group(decoder, group, optimizer, parameters)
                natural_hits += record["natural_hits"]
                hit_groups += record["natural_hits"] > 0
                nonzero_grad += record["policy_logprob_grad_norm"] > 0
                record.update(step=step, old_policy_step=start, group_size=GROUP_SIZE, sampled_size=SAMPLED,
                              forced_target_index=SAMPLED, user_id=train_users[step - 1]["user_id"],
                              sampled_hit_rate=natural_hits / (step * SAMPLED),
                              policy_gradient_groups=nonzero_grad, elapsed_seconds=perf_counter() - started,
                              **memory(decoder.device))
                stream.write(json.dumps({**record, "sampled_sids": group.codes[:SAMPLED],
                                         "forced_target_sid": group.codes[-1], "rewards": group.rewards.tolist(),
                                         "advantages": group.advantages.tolist(),
                                         "old_sequence_log_probs": group.old_log_probs.sum(dim=1).tolist()},
                                        allow_nan=False) + "\n")
                stream.flush()
                if step % 16 == 0 or step == len(samples):
                    log("gspo_progress", **record)
                if args.stage == "run" and step in steps:
                    destination = args.output / f"step-{step:04d}"
                    set_training_mode(decoder, False)
                    save_checkpoint(decoder, destination, identity, anchors, args)
                    set_training_mode(decoder, True)
                    paths[str(step)] = checkpoint_digest(destination)
                    log("gspo_checkpoint", step=step, checkpoint=str(destination))
    summary = {"groups": len(samples), "trajectories_per_group": GROUP_SIZE, "forced_target_groups": len(samples),
               "sampled_hit_groups": hit_groups, "sampled_hit_rate": natural_hits / (len(samples) * SAMPLED),
               "policy_gradient_groups": nonzero_grad, "seconds": perf_counter() - started,
               "checkpoints": paths, **memory(decoder.device)}
    write_json(args.output / "training-summary.json", summary)
    log("gspo_training_complete", **summary)
    if not nonzero_grad:
        raise RuntimeError("No group produced a nonzero GSPO policy gradient")
    return summary


def validate_candidate(args, data, inputs, checkpoint, fingerprint, users, protocol, step):
    directory = args.output / f"validation-{step:04d}"
    expected = {"step": step, "checkpoint_sha256": fingerprint, "validation_protocol": protocol,
                "scoring_backend": args.decode_backend, "decode_batch": args.decode_batch,
                "runtime": runtime_identity(args)}
    if directory.exists():
        result = read_json(directory / "metrics.json")
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Cached GSPO validation differs: {directory}")
        return result
    decoder = load_projection_checkpoint(args, data, checkpoint, inputs, users)
    log("gspo_validation_started", step=step, users_total=len(users))
    started = perf_counter()
    entries = list(ranking_batches(decoder, users, args.dataset, "validation", VALIDATION_K, args))
    metrics = validation_metrics(entries, users)
    result = {**expected, **metrics, "ndcg_at10": metrics["NDCG@10"], "checkpoint": checkpoint.name,
              "validation_seconds": perf_counter() - started}
    with staged_output(directory) as stage:
        write_json(stage / "metrics.json", result)
    log("gspo_validation", **result)
    return result


def select(args, data, identity, source, checkpoint, training, users):
    protocol = source["validation_protocol"]
    source_best = source["best_validation"]
    if args.reuse_sft:
        baseline = {**source_best, "step": 0, "validation_reused": True}
        log("sft_validation_reused", **baseline)
    else:
        baseline = validate_candidate(args, data, source["inputs"], checkpoint, source_best["checkpoint_sha256"],
                                      users, protocol, 0)
        release_device(args)
    records = [baseline]
    for step, digest in sorted(training["checkpoints"].items(), key=lambda pair: int(pair[0])):
        candidate = args.output / f"step-{int(step):04d}"
        if checkpoint_digest(candidate) != digest:
            raise ValueError(f"Invalid GSPO checkpoint for step {step}")
        records.append(validate_candidate(args, data, identity, candidate, digest, users, protocol, int(step)))
        release_device(args)
    best = max(records, key=lambda row: row["ndcg_at10"])
    selection = {"inputs": identity, "train_selection": TRAIN_SELECTION, "validation_protocol": protocol,
                 "best_validation": best, "candidates": records}
    write_json(args.output / "selection.json", selection)
    log("gspo_selected", step=best["step"], ndcg_at10=best["ndcg_at10"], source_ndcg_at10=baseline["ndcg_at10"])
    return selection


def source_data(args):
    run = read_json(args.sft_run / "run.json")
    source = read_json(args.sft_run / "selection.json")
    settings = run["arguments"]
    if (source["inputs"] != run["inputs"] or settings["stage"] != "run"
            or source["inputs"].get("method") != METHOD
            or source["inputs"].get("objective", "sft") != "sft" or source["train_selection"] != TRAIN_SELECTION):
        raise ValueError("Choose a completed SID projection SFT run")
    chosen = source["best_validation"]
    name = chosen["checkpoint"]
    if name != f"epoch-{chosen['epoch']:04d}" or not math.isfinite(chosen["ndcg_at10"]):
        raise ValueError("Invalid checkpoint selection in the source SFT run")
    checkpoint = args.sft_run / name
    if not (checkpoint / "sid_projection.pt").is_file() or checkpoint_digest(checkpoint) != chosen["checkpoint_sha256"]:
        raise ValueError("The selected SFT weights do not match their recorded fingerprint")
    for name in ("sid_data", "comparison_data", "cohort", "base_model"):
        setattr(args, name, Path(settings[name]))
    for name in ("dataset", "decode_backend", "decode_batch", "gradient_checkpointing"):
        setattr(args, name, settings[name])
    args.initialization = "sft"
    data = SequenceData(args.sid_data, args.comparison_data)
    users, cohort = load_cohort(args.cohort, args.dataset, data.users, data.item_ids, MAX_HISTORY)
    inputs = source["inputs"]
    if not inputs["smoke"]:
        require_formal_cohort(args.dataset, data, users, cohort)
    recorded = {"cohort": cohort, "comparison_data": data.comparison_identity,
                "sid_sha256": sha256(args.sid_data / "item_semantic_ids.jsonl"),
                "base_sha256": encoder_fingerprint(args.base_model),
                "decode_backend": args.decode_backend, "decode_batch": args.decode_batch}
    if any(inputs.get(key) != value for key, value in recorded.items()):
        raise ValueError("Base model, SID mapping, data or evaluation settings differ from SFT")
    valid = data.users
    if inputs["smoke"]:
        valid, users = sampled_users(valid, 8), sampled_users(users, 8)
        digest = hashlib.sha256(json.dumps([user["user_id"] for user in users]).encode()).hexdigest()
        cohort = {**cohort, "scope": "smoke", "cohort_sha256": digest}
    if source["validation_protocol"] != validation_record(valid, smoke=inputs["smoke"]):
        raise ValueError("SFT was selected with a different validation protocol")
    live_runtime = runtime_identity(args)
    args.reuse_sft = all(inputs.get(key) == value for key, value in live_runtime.items())
    identity = {**recorded, **live_runtime, "method": METHOD, "objective": "gspo", "smoke": inputs["smoke"],
                "source_sft_sha256": chosen["checkpoint_sha256"], "gspo_seed": SEED,
                "group_size": GROUP_SIZE, "rollout_groups": GROUPS_PER_ROLLOUT, "microbatch": MICROBATCH,
                "nll_weight": NLL_WEIGHT, "clip_low": CLIP_LOW, "clip_high": CLIP_HIGH, "lr_scale": LR_SCALE}
    return data, source, checkpoint, valid, users, cohort, identity


def execute(args):
    data, source, checkpoint, valid, users, cohort, identity = source_data(args)
    groups = args.groups if args.groups is not None else (16 if args.stage == "smoke" else 512)
    required = 16 if args.stage == "smoke" else 512
    if not 2 <= groups <= len(data.users) or (not identity["smoke"] and groups != required):
        raise ValueError(f"Use exactly {required} groups for this formal stage")
    train_users = random.Random(SEED).sample(data.users, groups)
    identity.update(groups=groups, train_users_sha256=hashlib.sha256(json.dumps(
        [row["user_id"] for row in train_users]).encode()).hexdigest())
    run_path = args.output / "run.json"
    if run_path.exists():
        if read_json(run_path)["inputs"] != identity:
            raise ValueError("This GSPO output directory was created with different settings")
    elif args.stage == "test":
        raise ValueError("Test needs a completed GSPO run and its selection")
    else:
        write_json(run_path, {"inputs": identity, "source_sft_run": str(args.sft_run),
                              "source_sft_epoch": source["best_validation"]["epoch"]})
    summary = args.output / "training-summary.json"
    selection_path = args.output / "selection.json"
    if args.stage != "test":
        if summary.exists():
            training = read_json(summary)
        else:
            train_log = args.output / "training.jsonl"
            if train_log.exists():
                raise ValueError("The GSPO training was interrupted. Start training in a fresh output directory")
            training = train(args, data, identity, source, checkpoint, train_users, valid)
            release_device(args)
        if not training["policy_gradient_groups"] or training["groups"] != groups:
            raise ValueError("GSPO training summary is incomplete or has no policy gradients")
        if args.stage == "smoke":
            return
        if set(training["checkpoints"]) != {str(groups // 2), str(groups)}:
            raise ValueError("Missing GSPO evaluation checkpoints")
        if not selection_path.exists():
            select(args, data, identity, source, checkpoint, training, valid)
    selection = read_json(selection_path)
    if selection["validation_protocol"] != source["validation_protocol"]:
        raise ValueError("The GSPO selection did not use the required validation cohort")
    winner = selection["best_validation"]["step"]
    details = {"objective": "gspo" if winner else "sft", "selected_gspo_step": winner,
               "gspo_training_groups": groups, "source_sft_epoch": source["best_validation"]["epoch"],
               "sft_validation_ndcg_at10": selection["candidates"][0]["ndcg_at10"]}
    test(args, data, identity, users, cohort, selection, valid, checkpoint=checkpoint if not winner else None,
         checkpoint_inputs=source["inputs"] if not winner else None, extra_details=details,
         prior_evaluation=args.sft_run / "evaluation.json" if not winner and args.reuse_sft else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "run", "test"))
    parser.add_argument("--sft-run", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--output", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groups", type=int, help="Override group count for a local smoke checkpoint")
    args = parser.parse_args()
    if args.output == args.sft_run or args.sft_run in args.output.parents:
        parser.error("Use an output directory separate from the SFT run")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or device.index not in (None, 0)):
        parser.error("Expose the assigned GPU as cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        execute(args)


if __name__ == "__main__":
    main()
