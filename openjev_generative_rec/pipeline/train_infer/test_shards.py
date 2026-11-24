"""Run formal OpenJev testing in independent shards, then merge on the CPU."""

import argparse
import json
import os
from pathlib import Path

from ..common import enable_comparison
from ..data_preparation.build_jev_data import SequenceData
from .recommend import add_scoring_backend, checkpoint_config, comparison_cohort, predictor, scoring_settings

enable_comparison()
from comparison.protocol import (SEED, TOP_K, checked_ranks, holdout, ranked_entry, read_json,
                                 staged_output, write_evaluation, write_json)


def shard_specs(directory, total, size, settings):
    return [(directory / f"shard-{index}", {**settings, "start": start, "stop": min(start + size, total)})
            for index, start in enumerate(range(0, total, size))]


def read_shard(directory, expected):
    path = directory / "rankings.json"
    if not path.is_file():
        raise ValueError(f"Missing test shard: {directory}")
    shard = read_json(path)
    if shard.get("metadata") != expected:
        raise ValueError(f"Test shard settings or user range differ: {directory}")
    return shard["entries"]


def run_shard(directory, metadata, users, catalog, make_predict):
    if directory.exists():
        checked_ranks(read_shard(directory, metadata), users, catalog)
        print(f"Test shard already complete: {directory}", flush=True)
        return

    predict = make_predict()
    entries = []
    for user in users:
        history, target = holdout(user, "test")
        entries.append(ranked_entry(user["user_id"], target, predict(history)))
        if len(entries) % 50 == 0:
            print(f"{directory.name}: {len(entries)}/{len(users)} users", flush=True)
    checked_ranks(entries, users, catalog)
    try:
        with staged_output(directory) as stage:
            write_json(stage / "rankings.json", {"metadata": metadata, "entries": entries})
    except OSError:
        # A retry may finish while another worker is publishing the same shard.
        if not directory.is_dir():
            raise
        checked_ranks(read_shard(directory, metadata), users, catalog)
    print(f"Test shard complete: {directory}", flush=True)


def merge_shards(output, specs, users, catalog, cohort, details, selection):
    expected = {directory for directory, _ in specs}
    found = set(specs[0][0].parent.glob("shard-*"))
    if missing := expected - found:
        raise ValueError(f"Missing test shards: {', '.join(sorted(path.name for path in missing))}")
    if found - expected:
        raise ValueError("Unexpected test shards; use the original shard size or a new directory")

    def entries():
        for directory, metadata in specs:
            rows = read_shard(directory, metadata)
            if len(rows) != metadata["stop"] - metadata["start"]:
                raise ValueError(f"Wrong number of users in {directory}")
            yield from rows

    return write_evaluation(output, entries(), users, catalog, cohort, "OpenJev", **details,
                            train_selection=selection["train_selection"],
                            validation_protocol=selection["validation_protocol"],
                            best_validation=selection["best_validation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--shard-index", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    merge = commands.add_parser("merge")
    merge.add_argument("--output", type=Path, required=True)
    for command in (infer, merge):
        for path in ("checkpoint", "base-model", "data", "comparison-data", "cohort", "shard-dir"):
            command.add_argument(f"--{path}", required=True, type=Path)
        command.add_argument("--shard-size", type=int, default=500)
        command.add_argument("--candidate-batch", type=int, default=4)
        add_scoring_backend(command)
    args = parser.parse_args()
    if min(args.shard_size, args.candidate_batch) < 1:
        parser.error("shard size and candidate batch must be positive")

    data = SequenceData(args.data, args.comparison_data)
    config = checkpoint_config(args.checkpoint, args.data / "item_semantic_ids.jsonl")
    users, cohort, selection = comparison_cohort(
        data, config, args.checkpoint, args.cohort, args.base_model,
        scoring_backend=args.scoring_backend, candidate_batch=args.candidate_batch)
    if selection is None or not users:
        parser.error("sharded testing requires a formal cohort and a selected checkpoint")
    args.beam_size = args.top_k = TOP_K
    args.batch_prefixes = True
    settings = {**cohort, "checkpoint_sha256": selection["best_validation"]["checkpoint_sha256"],
                "semantic_ids_sha256": config["semantic_ids_sha256"],
                "base_model": str(args.base_model.resolve(strict=True)),
                "beam_size": TOP_K, "batch_prefixes": True, "candidate_batch": args.candidate_batch,
                "policy_normalization": config.get("policy_normalization", "full"), "seed": SEED,
                **scoring_settings(args.scoring_backend)}
    specs = shard_specs(args.shard_dir, len(users), args.shard_size, settings)
    if args.command == "infer":
        if args.shard_index is None or not 0 <= args.shard_index < len(specs):
            parser.error(f"shard index must be between 0 and {len(specs) - 1}")
        directory, metadata = specs[args.shard_index]
        run_shard(directory, metadata, users[metadata["start"]:metadata["stop"]], data.item_ids,
                  lambda: predictor(args, data, config, cohort["dataset"]))
    else:
        details = {key: value for key, value in settings.items() if key not in cohort}
        details["protocol"] = "full catalog, seen items excluded, autoregressive beam, no teacher forcing"
        print(json.dumps(merge_shards(args.output, specs, users, data.item_ids, cohort, details, selection)), flush=True)


if __name__ == "__main__":
    main()
