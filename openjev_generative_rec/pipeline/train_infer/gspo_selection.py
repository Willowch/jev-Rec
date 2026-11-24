"""Validate SFT and GSPO checkpoints in independent GPU shards and publish the winner."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from ..common import enable_comparison, sha256
from ..data_preparation.build_jev_data import SequenceData
from .recommend import add_scoring_backend, scoring_settings

enable_comparison()
from comparison.protocol import (TRAIN_SELECTION, VALIDATION_K, dataset_identity, holdout, item_key,
                                 ranking_metrics, read_json, require_full_validation,
                                 require_training_selection, staged_output, validation_record, write_json)


def candidate_steps(steps):
    return steps // 2, steps


def candidate_path(run, step):
    return run / "candidates" / f"step-{step}"


def checkpoint_digest(checkpoint):
    digest = hashlib.sha256()
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file() and path != checkpoint / "selection.json":
            name = path.relative_to(checkpoint).as_posix()
            digest.update(json.dumps([name, sha256(path)]).encode())
    return digest.hexdigest()


def candidate_fingerprints(run, steps):
    return {str(step): checkpoint_digest(candidate_path(run, step)) for step in (0, *candidate_steps(steps))}


def require_sft_source(checkpoint, data):
    path = checkpoint.parent / "run.json"
    if checkpoint.name != "checkpoint" or not path.is_file():
        raise ValueError("Formal GSPO needs a completed single-target SFT run as its source")
    record = read_json(path)
    require_training_selection(record)
    source = record.get("data_manifest", {})
    identity = record.get("comparison_identity") or source.get("comparison_identity")
    if (record.get("status") != "complete" or record.get("arguments", {}).get("objective") != "sft"
            or record.get("train_positions") != data.train_positions
            or (identity is not None and identity != data.comparison_identity)
            or any(source.get(key) != data.manifest[key] for key in ("ratings_sha256", "semantic_ids_sha256"))):
        raise ValueError("The SFT run does not match the shared GSPO training data")


def read_candidate_run(run, data):
    record = read_json(run / "run.json")
    require_training_selection(record)
    args = record.get("arguments", {})
    if (record.get("status") != "complete" or args.get("objective") != "gspo"
            or not args.get("full_validation")
            or record.get("comparison_identity") != data.comparison_identity
            or record["data_manifest"]["semantic_ids_sha256"] != data.manifest["semantic_ids_sha256"]):
        raise ValueError("Use a completed GSPO --full-validation run with its original shared data and SID mapping")
    steps = candidate_steps(args["steps"])
    if record.get("includes_sft_candidate"):
        steps = (0, *steps)
    if set(record.get("validation_candidates", {})) != set(map(str, steps)):
        raise ValueError("The GSPO run is missing validation candidates")
    return record


def checked_candidate(run, step, record):
    checkpoint = candidate_path(run, step)
    digest = record["validation_candidates"].get(str(step))
    if digest is None or not checkpoint.is_dir() or checkpoint_digest(checkpoint) != digest:
        raise ValueError(f"Step {step} is not an intact validation candidate")
    return checkpoint, digest


def base_model_settings(path):
    base_model = Path(path).resolve(strict=True)
    if not (base_model / "config.json").is_file():
        raise ValueError(f"Formal evaluation requires a local base model directory: {base_model}")
    # Check the base model file inventory without rereading the multi-GB weights per shard.
    files = {}
    for path in sorted(base_model.iterdir()):
        if path.is_file():
            stat = path.stat()
            files[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"base_model": str(base_model), "base_model_files": files}


def validation_settings(args, record, protocol):
    base_model = args.base_model or record["arguments"]["base_model"]
    return {"version": 1, **base_model_settings(base_model), "candidate_batch": args.candidate_batch,
            "shard_size": args.shard_size,
            "validation_protocol": protocol, "beam_size": VALIDATION_K, "batch_prefixes": True,
            "candidates": record["validation_candidates"],
            **scoring_settings(getattr(args, "scoring_backend", "batched"))}


def scoring_audit_identity(data, checkpoint_sha256, base_model, candidate_batch):
    return {"comparison_identity": data.comparison_identity,
            "semantic_ids_sha256": data.manifest["semantic_ids_sha256"],
            "checkpoint_sha256": checkpoint_sha256, "candidate_batch": candidate_batch,
            **base_model_settings(base_model), **scoring_settings("prefix-cache")}


def require_scoring_audits(directory, data, settings):
    if directory is None:
        raise ValueError("Cached validation requires --backend-audits with a passing report for every candidate")
    for step, digest in settings["candidates"].items():
        path = directory / f"step-{step}.json"
        if not path.is_file():
            raise ValueError(f"Missing prefix-cache audit: {path}")
        audit = read_json(path).get("scoring_audit", {})
        expected = scoring_audit_identity(data, digest, settings["base_model"], settings["candidate_batch"])
        if audit.get("identity") != expected or audit.get("passed") is not True:
            raise ValueError(f"Prefix-cache audit failed or used different settings: {path}")


def check_validation_settings(directory, expected, *, create=False):
    destination = directory / "settings"
    path = destination / "validation.json"
    if create and not destination.exists():
        if any(directory.glob("step-*")) and not path.is_file():
            raise ValueError(f"Unverified validation shards in {directory}; choose a new --validation-dir")
        try:
            with staged_output(destination) as stage:
                write_json(stage / path.name, expected)
        except OSError:
            if not path.is_file():
                raise
    if not path.is_file() or read_json(path) != expected:
        raise ValueError(f"Missing or different validation settings in {directory}; use the original settings "
                         "or choose a new --validation-dir and regenerate all shards")


def shard_path(directory, step, index):
    return directory / f"step-{step}" / f"shard-{index}"


def shard_metadata(step, digest, settings, start, stop):
    settings_digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return {"step": step, "checkpoint_sha256": digest, "settings_sha256": settings_digest,
            "start": start, "stop": stop}


def checked_shard(directory, metadata, users):
    path = directory / "ranks.json"
    if not path.is_file():
        raise ValueError(f"Missing validation shard: {directory}")
    shard = read_json(path)
    if shard.get("metadata") != metadata or len(shard.get("ranks", [])) != len(users):
        raise ValueError(f"Validation shard does not match its settings, candidate or range: {directory}; "
                         "choose a new --validation-dir to regenerate it")
    ranks = []
    for (uid, target, rank), user in zip(shard["ranks"], users, strict=True):
        if ((uid, target) != (user["user_id"], user["items"][-2])
                or (rank is not None and (type(rank) is not int or not 1 <= rank <= VALIDATION_K))):
            raise ValueError(f"Validation shard has an invalid user, target or rank: {directory}")
        ranks.append(rank)
    return ranks


def validate_shard(args, data, record, settings):
    from ..models.openjev_runtime import load_model
    from .recommend import recommend

    checkpoint, digest = checked_candidate(args.run, args.step, record)
    start = args.shard_index * args.shard_size
    if start >= len(data.users):
        raise ValueError("Validation shard index is outside the dataset")
    stop = min(start + args.shard_size, len(data.users))
    check_validation_settings(args.validation_dir, settings, create=True)
    destination = shard_path(args.validation_dir, args.step, args.shard_index)
    metadata = shard_metadata(args.step, digest, settings, start, stop)
    users = data.users[start:stop]
    if destination.exists():
        checked_shard(destination, metadata, users)
        print(f"Validation shard already complete: {destination}", flush=True)
        return

    config = read_json(checkpoint / "recommendation.json")
    temperature = read_json(checkpoint / "temperature.json")["temperature"]
    model = load_model(checkpoint, settings["base_model"],
                       item_catalog=args.data / "item_semantic_ids.jsonl")
    ranks = []
    dataset = data.comparison_identity["dataset"]
    for user in users:
        history, target = holdout(user, "validation")
        results = recommend(model, data.item_ids, history, max_history=config["max_history"],
                            beam_size=VALIDATION_K, top_k=VALIDATION_K, candidate_batch=args.candidate_batch,
                            temperature=temperature, normalization=config["policy_normalization"],
                            batch_prefixes=True, scoring_backend=settings.get("scoring_backend", "batched"),
                            catalog_key=lambda item: item_key(dataset, item),
                            catalog_policy=data.catalog)
        rank = next((i for i, row in enumerate(results, 1) if row["item_id"] == target), None)
        ranks.append([user["user_id"], target, rank])
        if len(ranks) % 50 == 0:
            print(f"Step {args.step} shard {args.shard_index}: {len(ranks)}/{len(users)} users", flush=True)
    try:
        with staged_output(destination) as stage:
            write_json(stage / "ranks.json", {"metadata": metadata, "ranks": ranks})
    except OSError:
        if not destination.is_dir():
            raise
        checked_shard(destination, metadata, users)
    print(f"Validation shard complete: {destination}", flush=True)


def select_checkpoint(args, data, record, settings):
    if not data.users:
        raise ValueError("Full GSPO selection requires validation users")
    check_validation_settings(args.validation_dir, settings)
    reports = []
    for step in sorted(map(int, record["validation_candidates"])):
        _, digest = checked_candidate(args.run, step, record)
        ranks = []
        expected_shards = set()
        for index, start in enumerate(range(0, len(data.users), args.shard_size)):
            stop = min(start + args.shard_size, len(data.users))
            directory = shard_path(args.validation_dir, step, index)
            expected_shards.add(directory)
            metadata = shard_metadata(step, digest, settings, start, stop)
            ranks.extend(checked_shard(directory, metadata, data.users[start:stop]))
        if set((args.validation_dir / f"step-{step}").glob("shard-*")) != expected_shards:
            raise ValueError(f"Unexpected or duplicate validation shards for step {step}")
        reports.append({"step": step, "objective": "sft" if step == 0 else "gspo", "checkpoint_sha256": digest,
                        **ranking_metrics(ranks, (VALIDATION_K,))})
    if reports[0]["step"] == 0:
        baseline = reports[0]["NDCG@10"]
        for report in reports:
            report["ndcg_at10_delta_from_sft"] = report["NDCG@10"] - baseline
    best = max(reports, key=lambda report: (report["NDCG@10"], -report["step"]))
    checkpoint = args.output or args.run / "checkpoint"
    with staged_output(checkpoint) as stage:
        shutil.copytree(candidate_path(args.run, best["step"]), stage, dirs_exist_ok=True)
        if checkpoint_digest(stage) != best["checkpoint_sha256"]:
            raise ValueError("The selected checkpoint changed during publication")
        write_json(stage / "selection.json", {
            "objective": best["objective"], "train_selection": TRAIN_SELECTION, "comparison_identity": data.comparison_identity,
            "validation_protocol": settings["validation_protocol"], "validation_settings": settings,
            "best_validation": best, "candidates": reports,
        })
    return best


def require_selected_checkpoint(checkpoint, data):
    path = checkpoint / "selection.json"
    if not path.is_file():
        raise ValueError("Formal OpenJev testing requires the checkpoint published by gspo_selection select")
    record = read_json(path)
    settings = record.get("validation_settings", {})
    if settings.get("version") != 1:
        raise ValueError("Selected checkpoint lacks verified validation settings. Regenerate shards in a new "
                         "--validation-dir and publish with gspo_selection select --output")
    require_training_selection(record)
    require_full_validation(record.get("validation_protocol"), data.users)
    identity = data.comparison_identity or dataset_identity(
        data.manifest.get("dataset", "ml1m"), data.users, data.item_ids, data.manifest["max_history"])
    best = record["best_validation"]
    objective = "sft" if best["step"] == 0 else "gspo"
    if (record.get("objective") != objective or record.get("comparison_identity") != identity
            or settings["validation_protocol"] != record["validation_protocol"]
            or settings["candidates"].get(str(best["step"])) != best["checkpoint_sha256"]
            or best not in record.get("candidates", []) or best["checkpoint_sha256"] != checkpoint_digest(checkpoint)):
        raise ValueError("The selected weights or comparison data differ from the full validation run")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--step", type=int, required=True)
    validate.add_argument("--shard-index", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"),
                          help="Zero-based shard index, defaults to SLURM_ARRAY_TASK_ID")
    select = commands.add_parser("select")
    select.add_argument("--output", type=Path, help="Publish a new selected checkpoint; defaults to RUN/checkpoint")
    for command in (validate, select):
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--comparison-data", type=Path, required=True)
        command.add_argument("--shard-size", type=int, default=1000)
        command.add_argument("--validation-dir", type=Path, help="Shard directory; defaults to RUN/validation")
        command.add_argument("--base-model", help="Local base model; defaults to the training run's base model")
        command.add_argument("--candidate-batch", type=int, default=4)
        add_scoring_backend(command)
        command.add_argument("--backend-audits", type=Path, help="Directory of step-N.json parity reports")
    args = parser.parse_args()
    if min(args.shard_size, args.candidate_batch) < 1 or (args.command == "validate"
                                                        and (args.shard_index is None or args.shard_index < 0)):
        parser.error("Use a positive shard size and candidate batch, and a nonnegative shard index")
    args.run = args.run.resolve()
    args.validation_dir = (args.validation_dir or args.run / "validation").resolve()
    data = SequenceData(args.data, args.comparison_data)
    if sha256(args.data / "item_semantic_ids.jsonl") != data.manifest["semantic_ids_sha256"]:
        raise ValueError("The SID mapping differs from the preparation manifest")
    record = read_candidate_run(args.run, data)
    settings = validation_settings(args, record, validation_record(data.users))
    if args.scoring_backend == "prefix-cache":
        require_scoring_audits(args.backend_audits, data, settings)
    if args.command == "validate":
        validate_shard(args, data, record, settings)
    else:
        print(json.dumps(select_checkpoint(args, data, record, settings)), flush=True)


if __name__ == "__main__":
    main()
