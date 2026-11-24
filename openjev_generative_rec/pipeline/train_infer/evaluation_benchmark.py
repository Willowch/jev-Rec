"""Measure the current formal beam search on a small sample and estimate GPU time."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from ..common import enable_comparison
from ..data_preparation.build_jev_data import SequenceData
from .gspo_selection import checkpoint_digest, scoring_audit_identity
from .recommend import add_scoring_backend, checkpoint_config, predictor

enable_comparison()
from comparison.protocol import (SPLIT_PROTOCOL, TOP_K, VALIDATION_K, holdout, load_cohort, sampled_users,
                                 validation_users, write_json)


AUDIT_USERS = 32
MAX_LOGIT_ERROR = 0.05
MAX_PROBABILITY_ERROR = 0.005


def measure(predict, users, split, beam, count, warmup, synchronize):
    sample = sampled_users(users, count + warmup)
    if len(sample) != count + warmup:
        raise ValueError(f"{split} needs at least {count + warmup} users for measurement and warmup")

    def run(user):
        history, _ = holdout(user, split)
        predict(history, beam_size=beam, top_k=beam)
        synchronize()

    for user in sample[:warmup]:
        run(user)
    elapsed = []
    for user in sample[warmup:]:
        started = time.perf_counter()
        run(user)
        elapsed.append(time.perf_counter() - started)
    return {"users_measured": count, "warmup_users": warmup, "beam_size": beam,
            "mean_seconds_per_user": statistics.mean(elapsed),
            "p50_seconds_per_user": statistics.median(elapsed),
            "p90_seconds_per_user": sorted(elapsed)[math.ceil(0.90 * count) - 1]}


def estimate(measured, total, shard_size, setup_seconds):
    shards = math.ceil(total / shard_size)
    return {"users": total, "shards": shards,
            "estimated_gpu_hours": (total * measured["mean_seconds_per_user"] + shards * setup_seconds) / 3600,
            "estimated_p90_shard_hours": (min(total, shard_size) * measured["p90_seconds_per_user"]
                                          + setup_seconds) / 3600}


def compare_scores(reference, cached, temperature):
    if reference.keys() != cached.keys():
        return None
    logit_error = probability_error = 0.0
    for prefix, (options, logits) in reference.items():
        other_options, other_logits = cached[prefix]
        if options != other_options:
            return None
        delta = (logits - other_logits).abs().max().item()
        probabilities = (logits / temperature).softmax(-1)
        other_probabilities = (other_logits / temperature).softmax(-1)
        probability_delta = (probabilities - other_probabilities).abs().max().item()
        if not math.isfinite(delta) or not math.isfinite(probability_delta):
            return None
        logit_error = max(logit_error, delta)
        probability_error = max(probability_error, probability_delta)
    return logit_error, probability_error


def audit_beam(predict, users, beam, count, warmup, temperature):
    sample = sampled_users(users, count + warmup)
    if len(sample) != count + warmup:
        raise ValueError("Not enough validation users for the scoring audit")

    backends = ("batched", "prefix-cache")
    for user in sample[:warmup]:
        history, _ = holdout(user, "validation")
        for backend in backends:
            predict(history, beam_size=beam, top_k=beam, scoring_backend=backend)
            torch.cuda.synchronize()

    timings = {backend: [] for backend in backends}
    peaks = {backend: 0 for backend in backends}
    ranking_mismatches = rank_mismatches = trace_mismatches = 0
    logit_error = probability_error = 0.0
    differences = []
    for index, user in enumerate(sample[warmup:]):
        history, target = holdout(user, "validation")
        recommendations, traces = {}, {}
        # Alternate order so cold inputs or contention do not always favor one scorer.
        for backend in backends[index % 2:] + backends[:index % 2]:
            trace = {}
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            rows = predict(history, beam_size=beam, top_k=beam, scoring_backend=backend,
                           score_trace=lambda prefix, options, scores: trace.update({prefix: (options, scores)}))
            torch.cuda.synchronize()
            timings[backend].append(time.perf_counter() - started)
            peaks[backend] = max(peaks[backend], torch.cuda.max_memory_allocated())
            recommendations[backend] = [row["item_id"] for row in rows]
            traces[backend] = trace

        original, cached = (recommendations[backend] for backend in backends)
        ranking_differs = original != cached
        rank_differs = ((original.index(target) if target in original else None) !=
                        (cached.index(target) if target in cached else None))
        errors = compare_scores(traces["batched"], traces["prefix-cache"], temperature)
        ranking_mismatches += ranking_differs
        rank_mismatches += rank_differs
        trace_mismatches += errors is None
        if errors is not None:
            logit_error = max(logit_error, errors[0])
            probability_error = max(probability_error, errors[1])
        if ranking_differs or rank_differs or errors is None:
            differences.append(user["user_id"])

    performance = {}
    for backend in backends:
        performance[backend] = {"mean_seconds_per_user": statistics.mean(timings[backend]),
                                "p90_seconds_per_user": sorted(timings[backend])[math.ceil(0.90 * count) - 1],
                                "peak_gpu_gib": peaks[backend] / 2**30}
    speedup = (performance["batched"]["mean_seconds_per_user"] /
               performance["prefix-cache"]["mean_seconds_per_user"])
    return {"beam_size": beam, "users_measured": count, "ranking_mismatches": ranking_mismatches,
            "target_rank_mismatches": rank_mismatches, "trace_mismatches": trace_mismatches,
            "mismatched_users": differences, "max_logit_abs_error": logit_error,
            "max_probability_abs_error": probability_error, "speedup": speedup, "performance": performance,
            "passed": (not (ranking_mismatches or rank_mismatches or trace_mismatches)
                       and logit_error <= MAX_LOGIT_ERROR and probability_error <= MAX_PROBABILITY_ERROR
                       and speedup > 1.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "base-model", "data", "comparison-data", "cohort", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--users", type=int, help="Measured users per mode (30, or 32 in audit mode)")
    parser.add_argument("--warmup-users", type=int, default=2)
    parser.add_argument("--candidate-batch", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--validation-candidates", type=int, default=3)
    add_scoring_backend(parser)
    parser.add_argument("--compare-backends", action="store_true", help="Audit both beams on validation users only")
    args = parser.parse_args()
    if args.users is None:
        args.users = AUDIT_USERS if args.compare_backends else 30
    if min(args.users, args.candidate_batch, args.shard_size, args.validation_candidates) < 1 or args.warmup_users < 0:
        parser.error("Use positive counts, with a nonnegative warmup count")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        parser.error("Run the benchmark on the CUDA GPU that will run formal evaluation")

    started = time.perf_counter()
    data = SequenceData(args.data, args.comparison_data)
    config = checkpoint_config(args.checkpoint, args.data / "item_semantic_ids.jsonl")
    if data.manifest["split_protocol"] != SPLIT_PROTOCOL or config["max_history"] != data.manifest["max_history"]:
        raise ValueError("Checkpoint history or data split differs from the formal protocol")
    dataset = data.comparison_identity["dataset"]
    selected, cohort = load_cohort(args.cohort, dataset, data.users, data.item_ids, config["max_history"])
    if cohort["scope"] not in ("formal", "pilot"):
        parser.error("Benchmark estimates require a formal or pilot test cohort")
    args.batch_prefixes = True
    predict = predictor(args, data, config, dataset)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started

    report = {"dataset": dataset, "gpu": torch.cuda.get_device_name(), "checkpoint": str(args.checkpoint.resolve()),
              "base_model": str(args.base_model.resolve()), "cohort_sha256": cohort["cohort_sha256"],
              "candidate_batch": args.candidate_batch,
              "shard_size": args.shard_size, "setup_seconds_per_shard": setup_seconds,
              "validation_candidates": args.validation_candidates}
    if args.compare_backends:
        temperature = json.loads((args.checkpoint / "temperature.json").read_text())["temperature"]
        beams = [audit_beam(predict, data.users, beam, args.users, args.warmup_users, temperature)
                 for beam in (VALIDATION_K, TOP_K)]
        report["scoring_audit"] = {
            "identity": scoring_audit_identity(data, checkpoint_digest(args.checkpoint),
                                               args.base_model, args.candidate_batch),
            "split": "validation", "max_allowed_logit_error": MAX_LOGIT_ERROR,
            "max_allowed_probability_error": MAX_PROBABILITY_ERROR, "beams": beams,
            "passed": args.users >= AUDIT_USERS and all(result["passed"] for result in beams),
        }
    else:
        report["scoring_backend"] = args.scoring_backend
        valid = validation_users(data.users, pilot=cohort["scope"] == "pilot")
        for split, beam, users in (("validation", VALIDATION_K, valid), ("test", TOP_K, selected)):
            torch.cuda.reset_peak_memory_stats()
            result = measure(predict, users, split, beam, args.users, args.warmup_users, torch.cuda.synchronize)
            report[split] = {**result, **estimate(result, len(users), args.shard_size, setup_seconds),
                             "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
        report["estimated_serial_gpu_hours"] = (report["validation"]["estimated_gpu_hours"] * args.validation_candidates
                                                + report["test"]["estimated_gpu_hours"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
