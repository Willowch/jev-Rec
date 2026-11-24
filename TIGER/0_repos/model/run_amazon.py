"""Prepare, train and evaluate TIGER on an Amazon 5-core category."""

import argparse
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from uuid import uuid4

import torch

from tiger_model import TIGER, TIGER_MODEL_DEFAULTS, set_seed, train
from experiment_tracking import PROJECT_ROOT, sha256_file, write_json
from amazon_data import DATASETS, load_data, prepare_data
from amazon_eval import Catalog, calibrate, evaluate_split
from dataloader import GenRecDataLoader
from ml1m_data import CODEBOOK_SIZE, SID_LAYERS, TrainingPairs, load_native_codes
from ml1m_eval import comparison_protocol
from run_ml1m import DEFAULT_SENTENCE_MODEL


def positive(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def user_count(value):
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("must be nonnegative (0 means all)")
    return count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, purpose in (
        ("all", "Run every stage in a new run directory"),
        ("prepare", "Build chronological sequences and the product catalog"),
        ("sid", "Encode catalog text and train/export native RQ-VAE SIDs"),
        ("train", "Train TIGER and select a checkpoint on validation NDCG"),
        ("evaluate", "Use the selected checkpoint to recommend and score the test set"),
    ):
        command = stages.add_parser(stage, help=purpose, description=purpose)
        command.add_argument("--dataset", choices=DATASETS, required=True)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--seed", type=int, help="Defaults to 42 for a new run")
        if stage in ("all", "prepare"):
            command.add_argument("--raw-dir", type=Path, help="Directory containing both gzip files")
            command.add_argument("--comparison-data", type=Path, help="Canonical OpenJev comparison dataset")
            command.add_argument("--max-history", type=positive, default=20)
            command.add_argument("--min-rating", type=float, default=0.0)
            command.add_argument("--smoke", action="store_true", help="Use a deterministic user subset")
            command.add_argument("--smoke-users", type=positive, default=256)
        if stage != "prepare":
            command.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        if stage in ("all", "sid"):
            command.add_argument("--sentence-model", type=Path, default=DEFAULT_SENTENCE_MODEL)
            command.add_argument("--embedding-batch-size", type=positive, default=32)
            command.add_argument("--rqvae-epochs", type=positive)
        if stage in ("all", "train"):
            command.add_argument("--tiger-epochs", type=positive)
            command.add_argument("--batch-size", type=positive)
            command.add_argument("--max-train-steps", type=positive, help="Optimizer updates per epoch")
            command.add_argument("--eval-step", type=positive, default=5)
            command.add_argument("--early-stop", type=positive, default=30)
            command.add_argument("--calibration-rows", type=positive)
            command.add_argument("--validation-users", type=user_count, help="0 means all users")
        if stage in ("all", "train", "evaluate"):
            command.add_argument("--eval-batch-size", type=positive)
            command.add_argument("--beam-size", type=positive)
            command.add_argument("--top-k", type=positive)
        if stage in ("all", "evaluate"):
            cohort_group = command.add_mutually_exclusive_group()
            cohort_group.add_argument("--test-users", type=user_count, help="0 means all users")
            cohort_group.add_argument("--cohort", type=Path, help="Fixed comparison test cohort")
        if stage == "evaluate":
            command.add_argument("--evaluation-dir", type=Path, help="New output directory for a rerun")
    args = parser.parse_args()
    if getattr(args, "cohort", None) and any(value not in (None, 20) for value in (args.beam_size, args.top_k)):
        parser.error("--cohort requires test evaluation with beam=20 and top-k=20")
    if args.stage == "all" and args.cohort and args.raw_dir and not args.comparison_data:
        parser.error("--cohort requires shared comparison data for a new run")
    if args.stage == "all" and not args.smoke and not args.cohort and (args.comparison_data or not args.raw_dir):
        parser.error("Formal shared-data runs require a full test cohort")
    return args


def fill_defaults(args, defaults):
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


@contextmanager
def new_stage(directory):
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    except BaseException:
        failed = directory.with_name(f"{directory.name}.failed-{uuid4().hex[:8]}")
        directory.rename(failed)
        print(f"Incomplete stage saved at {failed}", file=sys.stderr, flush=True)
        raise


def encode_catalog(catalog, model_path, device, batch_size, output):
    comparison_protocol()
    from comparison.features import encode_sentence_t5

    encode_sentence_t5(catalog, "amazon", model_path, device, batch_size, output)


def run_rqvae(script, options, log):
    command = [sys.executable, "-B", "-u", str(PROJECT_ROOT / "0_repos/rqvae" / script), *options]
    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if script == "generate_code.py":
        # This subprocess reads the checkpoint produced locally by the preceding training command.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print(f"Running RQ-VAE {script}. Log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"RQ-VAE {script} exited with {result.returncode}. See {log.name}")


def build_sids(args, directory, data, catalog, device):
    if not (args.sentence_model / "modules.json").is_file():
        raise FileNotFoundError(f"Local Sentence-T5 snapshot is missing: {args.sentence_model}")
    if len(catalog) < CODEBOOK_SIZE:
        raise ValueError(f"RQ-VAE needs at least {CODEBOOK_SIZE} catalog items for K-means. "
                         "Increase --smoke-users or check the raw data.")
    epochs = args.rqvae_epochs or (1 if data.manifest["smoke"] else 3000)
    embedding = directory / "item_emb.parquet"
    codes_path = directory / "item_sid.npy"
    run_root = directory / "rqvae"
    feature_reference = None
    if reference := data.manifest.get("comparison_data"):
        from comparison.features import sentence_t5
        from comparison.prepare import load_prepared

        shared = (args.run_dir / "data" / reference).resolve()
        _, items, prepared = load_prepared(shared, args.dataset)
        embedding, _ = sentence_t5(shared, args.dataset, items, prepared, args.sentence_model,
                                   device, args.embedding_batch_size)
        feature_reference = os.path.relpath(embedding, directory)
    else:
        encode_catalog(catalog, args.sentence_model.resolve(), device, args.embedding_batch_size, embedding)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    options = ["--data_path", str(embedding), "--dataset", args.dataset, "--disable_mlflow",
               "--run_root", str(run_root), "--device", args.device, "--num_workers", "0",
               "--seed", str(args.seed), "--epochs", str(epochs)]
    if data.manifest["smoke"]:
        options += ["--warmup_epochs", "0"]
    run_rqvae("main.py", options, directory / "rqvae_train.log")
    (summary,) = run_root.glob(f"{args.dataset}/stage1_rqvae/*/summary.json")
    checkpoint = read_json(summary)["checkpoint_path"]
    run_rqvae("generate_code.py",
              ["--ckpt_path", checkpoint, "--data_path", str(embedding), "--output_file", str(codes_path),
               "--device", args.device, "--disable_mlflow"], directory / "rqvae_export.log")
    load_native_codes(codes_path, embedding, catalog)
    write_json(directory / "manifest.json", {
        "dataset": args.dataset, "catalog_sha256": data.manifest["catalog_sha256"],
        "embedding_sha256": sha256_file(embedding), "sid_sha256": sha256_file(codes_path),
        "embedding_ref": feature_reference,
        "sentence_model": args.sentence_model.resolve(), "embedding_batch_size": args.embedding_batch_size,
        "rqvae_epochs": epochs, "seed": args.seed,
        "sid_layers": SID_LAYERS, "classes_per_layer": CODEBOOK_SIZE,
    })


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog(directory, data, items):
    manifest = read_json(directory / "manifest.json")
    if manifest["catalog_sha256"] != data.manifest["catalog_sha256"]:
        raise ValueError("The prepared catalog does not match the SID stage")
    embedding = (directory / (manifest.get("embedding_ref") or "item_emb.parquet")).resolve()
    codes_path = directory / "item_sid.npy"
    for path, key in ((embedding, "embedding_sha256"), (codes_path, "sid_sha256")):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"{path.name} differs from the SID manifest")
    return Catalog(load_native_codes(codes_path, embedding, items), data.manifest["dataset"]), manifest["sid_sha256"]


def train_tiger(data, catalog, sid_sha256, args, device, directory):
    smoke = data.manifest["smoke"]
    shared = "comparison_identity" in data.manifest
    if shared and args.validation_users is not None:
        raise ValueError("Shared-data training selects its own fixed validation users")
    fill_defaults(args, {
        "tiger_epochs": 1 if smoke else 200, "batch_size": 16 if smoke else 256,
        "max_train_steps": 3 if smoke else None, "calibration_rows": 3 if smoke else 30,
        "validation_users": 1 if smoke else 0, "eval_batch_size": 16,
        "beam_size": 3 if smoke else 10, "top_k": 3 if smoke else 10,
    })
    validation = copy(args)
    selected, selection = None, None
    if shared:
        protocol = comparison_protocol()
        selected = protocol.validation_users(data.users, smoke=smoke)
        selection = protocol.validation_record(selected, smoke=smoke)
        validation.beam_size = validation.top_k = protocol.VALIDATION_K
    set_seed(args.seed)
    model = TIGER(TIGER_MODEL_DEFAULTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loader = GenRecDataLoader(TrainingPairs(data, catalog.tokens), batch_size=args.batch_size,
                              num_workers=0)
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir()
    best_score = -math.inf
    optimizer_steps = 0
    epochs = []
    for epoch in range(1, args.tiger_epochs + 1):
        loss = train(model, loader, optimizer, device, max_steps=args.max_train_steps)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite TIGER loss at epoch {epoch}")
        optimizer_steps += min(len(loader), args.max_train_steps or len(loader))
        metrics = {"epoch": epoch, "loss": loss, "optimizer_steps": optimizer_steps}
        epochs.append(metrics)
        print(f"TIGER epoch {epoch}: loss={loss:.6f}, updates={optimizer_steps}", flush=True)
        if epoch % args.eval_step and epoch != args.tiger_epochs:
            continue

        model.eval()
        calibration = calibrate(model, data, catalog, args, device)
        candidate = None if shared else directory / "validation_candidate.json"
        valid_metrics = evaluate_split(model, data, catalog, "validation", args.validation_users,
                                       calibration["temperature"], validation, device, candidate, users=selected)
        score = valid_metrics[f"NDCG@{validation.top_k}"]
        metrics.update(validation_ndcg=score)
        print(f"Validation NDCG@{validation.top_k}={score:.6f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), checkpoint / "best_model.pth")
            write_json(checkpoint / "temperature.json", calibration)
            if candidate is None:
                write_json(directory / "validation.json", valid_metrics)
            else:
                candidate.replace(directory / "validation.json")
        else:
            if candidate is not None:
                candidate.unlink()
            if epoch - best_epoch >= args.early_stop:
                break
    write_json(directory / "summary.json", {
        "dataset": args.dataset, "configuration": vars(args), "model": TIGER_MODEL_DEFAULTS,
        "sequences_sha256": data.manifest["sequences_sha256"], "sid_sha256": sid_sha256,
        "max_history": data.manifest["max_history"], "train_pairs": data.train_pairs,
        "train_selection": comparison_protocol().TRAIN_SELECTION,
        "best_epoch": best_epoch, "epochs_trained": epoch, "optimizer_steps": optimizer_steps,
        "training": epochs, "validation_protocol": selection,
    })


def evaluate_checkpoint(data, catalog, sid_sha256, args, device, train_dir, output_dir, cohort=None):
    summary = read_json(train_dir / "summary.json")
    if (summary["dataset"] != args.dataset or summary["sid_sha256"] != sid_sha256
            or summary["sequences_sha256"] != data.manifest["sequences_sha256"]
            or summary["max_history"] != data.manifest["max_history"]):
        raise ValueError("The checkpoint, prepared data and SID stage do not match")
    selection = summary.get("validation_protocol")
    if not data.manifest["smoke"] and (cohort or "comparison_identity" in data.manifest):
        comparison_protocol().require_training_selection(summary)
        comparison_protocol().require_full_validation(selection, data.users)
    configuration = summary["configuration"]
    fill_defaults(args, {name: configuration[name]
                         for name in ("seed", "beam_size", "top_k", "eval_batch_size")})
    test_users = configuration.get("test_users")
    if test_users is None:
        test_users = 1 if data.manifest["smoke"] else 0
    fill_defaults(args, {"test_users": test_users})
    checkpoint = train_dir / "checkpoint"
    model = TIGER(summary["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint / "best_model.pth", map_location=device))
    model.eval()
    calibration = read_json(checkpoint / "temperature.json")
    output = output_dir / "evaluation.json"
    evaluation = copy(args)
    if cohort:
        evaluation.beam_size = evaluation.top_k = cohort[1]["top_k"]
    metrics = evaluate_split(model, data, catalog, "test", args.test_users,
                              calibration["temperature"], evaluation, device, output, cohort,
                               selection=selection, train_selection=summary.get("train_selection"))
    print(json.dumps(metrics), flush=True)
    print(f"Evaluation: {output}", flush=True)


def main():
    args = parse_args()
    run = args.run_dir.resolve()
    args.run_dir = run
    if args.stage == "all":
        run.mkdir(parents=True, exist_ok=False)
    if args.stage in ("all", "prepare"):
        if not math.isfinite(args.min_rating):
            raise ValueError("--min-rating must be finite")
        fill_defaults(args, {"seed": 42})
        protocol = comparison_protocol()
        from comparison.sources import raw_directory

        shared = args.comparison_data or protocol.DATA_ROOT / args.dataset
        if args.comparison_data is None and args.raw_dir is not None:
            shared = None
        raw = args.raw_dir or raw_directory(args.dataset)
        with new_stage(run / "data") as directory:
            prepare_data(args.dataset, raw, directory, max_history=args.max_history,
                         min_rating=args.min_rating, smoke=args.smoke, smoke_users=args.smoke_users,
                         seed=args.seed, comparison_data=shared)
    if args.stage == "prepare":
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or a CUDA-enabled TIGER environment.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    data, items = load_data(run / "data", args.dataset)
    cohort = (comparison_protocol().load_cohort(args.cohort, args.dataset, data.users,
                                                [row["item_id"] for row in items], data.manifest["max_history"])
              if getattr(args, "cohort", None) else None)
    if cohort and data.manifest["smoke"] != (cohort[1]["scope"] == "smoke"):
        raise ValueError("Match the TIGER training scope to the evaluation cohort")
    if args.stage == "evaluate" and not data.manifest["smoke"] and "comparison_identity" in data.manifest and not cohort:
        raise ValueError("Formal shared-data evaluation requires a full test cohort")
    if args.stage in ("all", "sid", "train"):
        fill_defaults(args, {"seed": data.manifest["seed"]})
    if args.stage in ("all", "sid"):
        with new_stage(run / "sid") as directory:
            build_sids(args, directory, data, items, device)
    if args.stage == "sid":
        return

    catalog, sid_sha256 = load_catalog(run / "sid", data, items)
    if args.stage in ("all", "train"):
        with new_stage(run / "tiger") as directory:
            train_tiger(data, catalog, sid_sha256, args, device, directory)
    if args.stage in ("all", "evaluate"):
        output_dir = getattr(args, "evaluation_dir", None) or run / "evaluation"
        with new_stage(output_dir.resolve()) as directory:
            evaluate_checkpoint(data, catalog, sid_sha256, args, device, run / "tiger", directory, cohort)


if __name__ == "__main__":
    main()
