"""Amazon 类目的数据准备、SFT、推理与离线评估入口。"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .common import ROOT, enable_comparison, sha256
from .data_preparation.prepare_amazon import DATASETS, SEED, prepare_dataset, staged_directory


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def test_user_count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("use 0 for all users or a positive sample size")
    return number


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("use a single run name such as sft or sft-1000")
    return value


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--comparison-data", type=Path, help="Shared sequences and metadata")
    parser.add_argument("--model-data", type=Path, help="Directory for this model's prepared SID artifacts")
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/qwen3.5-2b")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs/amazon")
    stages = parser.add_subparsers(dest="stage", required=True)
    prepare = stages.add_parser("prepare", help="create embeddings, semantic IDs and sequences")
    prepare.add_argument("--embedding-batch-size", type=positive_int, default=8)
    prepare.add_argument("--collision-policy", choices=("allow", "append"), default="allow")
    train = stages.add_parser("train", help="run layered-SID SFT with history memory")
    train.add_argument("--steps", type=positive_int)
    train.add_argument("--calibration-rows", type=positive_int)
    train.add_argument("--sft-recipe", choices=("legacy", "minionerec"), default="legacy")
    recommend = stages.add_parser("recommend", help="predict next items from an ordered ASIN history")
    recommend.add_argument("--history", nargs="+", required=True, metavar="ASIN")
    evaluate = stages.add_parser("evaluate", help="evaluate chronological held-out test targets")
    cohort_group = evaluate.add_mutually_exclusive_group()
    cohort_group.add_argument("--test-users", type=test_user_count, help="0 means all users")
    cohort_group.add_argument("--cohort", type=Path, help="fixed comparison cohort")
    for stage in (train, recommend, evaluate):
        stage.add_argument("--run-name", type=run_name, default="sft")
        stage.add_argument("--candidate-batch", type=positive_int, default=1)
        stage.add_argument("--seed", type=int, default=SEED)
    for stage in (recommend, evaluate):
        stage.add_argument("--beam-size", type=positive_int)
        stage.add_argument("--top-k", type=positive_int)
        stage.add_argument("--batch-prefixes", action="store_true", help="batch candidates across prefixes")
        stage.add_argument("--output", type=Path, help="JSON output path, relative to the project root")
    return parser


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    return path


def run_module(module, arguments):
    subprocess.run([sys.executable, "-u", "-m", module, *map(str, arguments)], cwd=ROOT, check=True)


def train_sft(args, sequences, stage_dir, manifest, shared):
    checkpoint = ROOT / "models/openjev-2b/package/checkpoint"
    for name in ("model.json", "head.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
        require_file(checkpoint / name)
    if sha256(sequences / "item_semantic_ids.jsonl") != manifest["semantic_ids_sha256"]:
        raise ValueError("prepared SID mapping does not match the sequence manifest")
    # Reserve the run directory before loading the model to exclude concurrent launches.
    stage_dir.mkdir(parents=True, exist_ok=False)
    steps = args.steps or (300 if args.sft_recipe == "minionerec" else 3)
    calibration_rows = args.calibration_rows or (30 if args.sft_recipe == "minionerec" else 3)
    run_module("pipeline.train_infer.train_lora", [
        *shared, "--checkpoint", checkpoint, "--output", stage_dir, "--objective", "sft",
        "--sid-encoding", "layered", "--history-memory", "on", "--steps", steps,
        "--calibration-rows", calibration_rows, "--sft-recipe", args.sft_recipe,
    ])


def run_recommendation(args, stage_dir, shared):
    summary = require_file(stage_dir / "summary.json")
    if json.loads(summary.read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError(f"training has not completed: {stage_dir}")
    checkpoint = stage_dir / "checkpoint"
    for name in ("recommendation.json", "temperature.json"):
        require_file(checkpoint / name)
    arguments = [*shared, "--checkpoint", checkpoint]
    for flag, value in (("--beam-size", args.beam_size), ("--top-k", args.top_k)):
        if value is not None:
            arguments.extend([flag, value])
    if args.batch_prefixes:
        arguments.append("--batch-prefixes")
    if args.stage == "recommend":
        arguments.extend(["--history", *args.history])
        filename = "recommendations.json"
    else:
        if args.cohort is not None:
            arguments.extend(["--cohort", args.cohort.resolve()])
        elif args.test_users is not None:
            arguments.extend(["--test-users", args.test_users])
        filename = "evaluation.json"
    output = ROOT / args.output if args.output is not None else stage_dir.parent / filename
    comparison = args.stage == "evaluate" and args.cohort is not None
    if comparison and output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(output.parent, ".openjev-output-") as temporary:
        staged = temporary / filename
        run_module("pipeline.train_infer.recommend", [*arguments, "--output", staged])
        if comparison:
            output.hardlink_to(staged)
        else:
            staged.replace(output)
    print(f"Saved {output}", flush=True)


def main():
    parser = cli_parser()
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    threads = os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)

    enable_comparison()
    from comparison.protocol import DATA_ROOT
    from comparison.sources import raw_directory

    raw = ROOT / "data" / args.dataset
    if not (raw / f"{args.dataset}_5.json.gz").is_file():
        raw = raw_directory(args.dataset)
    default_shared = DATA_ROOT / args.dataset
    shared_data = args.comparison_data or (default_shared if (default_shared / "manifest.json").is_file() else None)
    model_data = args.model_data or ROOT / "data" / args.dataset / "openjev"
    base_model = args.base_model.resolve()
    try:
        if args.stage == "prepare":
            prepare_dataset(args.dataset, raw, base_model, batch_size=args.embedding_batch_size,
                            comparison_data=shared_data, output=model_data.resolve(),
                            collision_policy=args.collision_policy)
            return

        sequences = (model_data / "sequences").resolve()
        manifest = json.loads(require_file(sequences / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("dataset") != args.dataset:
            raise ValueError(f"sequence manifest does not belong to {args.dataset}")
        for path in (sequences / "item_semantic_ids.jsonl", base_model / "config.json"):
            require_file(path)
        stage_dir = args.runs_root.resolve() / args.dataset / args.run_name / "stage1"
        shared = ["--data", sequences, "--base-model", base_model,
                  "--candidate-batch", args.candidate_batch, "--seed", args.seed]
        if shared_data is not None:
            shared.extend(["--comparison-data", shared_data.resolve()])
        if args.stage == "train":
            train_sft(args, sequences, stage_dir, manifest, shared)
        else:
            run_recommendation(args, stage_dir, shared)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, f"OpenJev {args.stage} failed (exit code {error.returncode})\n")


if __name__ == "__main__":
    main()
