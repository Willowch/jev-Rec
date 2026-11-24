"""Run the LETTER-TIGER baseline on the fixed OpenJev comparison datasets."""

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIGER_CODE = ROOT / "TIGER" / "0_repos"
sys.path[:0] = [str(ROOT), str(TIGER_CODE), str(TIGER_CODE / "model")]

from comparison.prepare import load_prepared
from comparison.protocol import (COMPARISON_DATASETS, MAX_HISTORY, PILOT_DATASET, SEED, TRAIN_SELECTION,
                                 VALIDATION_INTERVAL, load_cohort, read_json, require_full_validation,
                                 require_training_selection, sha256, staged_output, validation_record,
                                 validation_users, write_json)
from comparison.recbole_runner import choose_gpu


MODEL = "LETTER-TIGER"
UPSTREAM = "https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=COMPARISON_DATASETS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New JSON file for --evaluate-only")
    parser.add_argument("--sentence-model", type=Path, help="Path to the local TIGER Sentence-T5 snapshot")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-step", type=int, default=VALIDATION_INTERVAL, help="Epochs between validations")
    parser.add_argument("--cf-epochs", type=int, help="Collaborative encoder epochs in the pilot")
    parser.add_argument("--tokenizer-epochs", type=int, help="Tokenizer epochs in the pilot")
    parser.add_argument("--generator-epochs", type=int, help="Generator epochs in the pilot")
    parser.add_argument("--trials", type=int, choices=(1, 2), help="Generator trial count in the pilot")
    args = parser.parse_args()
    if args.device not in ("cpu", "cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdecimal()):
        parser.error("device must be cpu, cuda or cuda:N")
    if min(args.embedding_batch_size, args.train_batch_size, args.eval_batch_size, args.eval_step) < 1:
        parser.error("batch sizes and evaluation interval must be positive")
    if args.evaluate_only != (args.output is not None):
        parser.error("--evaluate-only and --output must be passed together")
    budgets = (args.cf_epochs, args.tokenizer_epochs, args.generator_epochs, args.trials)
    if any(value is not None for value in budgets) and args.dataset != PILOT_DATASET:
        parser.error("The budget overrides are for the MovieLens-100K pilot")
    if any(value is not None and value < 1 for value in budgets):
        parser.error("Pilot budgets must be positive")
    if args.dataset == PILOT_DATASET:
        args.cf_epochs = args.cf_epochs or 20
        args.tokenizer_epochs = args.tokenizer_epochs or 25
        args.generator_epochs = args.generator_epochs or 20
        args.trials = args.trials or 1
    return args


def load_catalog(path, items, dataset):
    from amazon_eval import Catalog
    from ml1m_data import CODEBOOK_SIZE, SID_LAYERS

    rows = read_json(path)
    if [row["item_id"] for row in rows] != items:
        raise ValueError("LETTER SID rows do not match the canonical catalog order")
    codes = [tuple(row["sid"]) for row in rows]
    if any(len(code) != SID_LAYERS or any(type(digit) is not int or not 0 <= digit < CODEBOOK_SIZE
                                         for digit in code) for code in codes):
        raise ValueError("Invalid LETTER SID codebook indices")
    if len(set(codes)) != len(codes):
        raise ValueError("LETTER SIDs must uniquely identify every catalog item")
    return Catalog(dict(zip(items, codes)), dataset)


def train_run(args, directory, data, metadata, device):
    from features import encode_text, train_cf
    from generative import train_generator
    from tiger_model import TIGER_MODEL_DEFAULTS, set_seed
    from run_ml1m import DEFAULT_SENTENCE_MODEL
    from tokenizer import train_tokenizer

    users, items, manifest = data
    args.sentence_model = (args.sentence_model or DEFAULT_SENTENCE_MODEL).resolve()
    set_seed(SEED)
    embeddings, text_source = encode_text(args, items, manifest, device)
    cf, support = train_cf(args, users, items, manifest, directory)
    set_seed(SEED)
    codes = train_tokenizer(embeddings, cf, support, args.smoke, device, directory / "tokenizer",
                            epochs=args.tokenizer_epochs)
    sid_file = directory / "sids.json"
    write_json(sid_file, [{"item_id": item, "sid": code} for item, code in zip(items, codes)])
    catalog = load_catalog(sid_file, items, args.dataset)
    trials, winner, selection = train_generator(args, users, catalog, directory, device)
    artifacts = (sid_file, directory / "cf" / "embeddings.npz", directory / "cf" / "sasrec.pth",
                 directory / "tokenizer" / "best.pth", directory / winner["trial"] / "best.pth")
    summary = {
        "model": MODEL, "upstream": UPSTREAM, "seed": SEED, "identity": manifest["identity"],
        "evaluation": metadata, "text_source": text_source, "model_config": TIGER_MODEL_DEFAULTS,
        "device": args.device, "embedding_batch_size": args.embedding_batch_size,
        "eval_batch_size": args.eval_batch_size, "eval_step": args.eval_step, "trials": trials, "winner": winner,
        "training_budget": {"cf_epochs": args.cf_epochs, "tokenizer_epochs": args.tokenizer_epochs,
                            "generator_epochs": args.generator_epochs, "generator_trials": args.trials},
        "validation_protocol": selection, "train_selection": TRAIN_SELECTION, "train_pairs": len(users),
        "artifacts": {path.relative_to(directory).as_posix(): sha256(path) for path in artifacts},
        "versions": {name: version(name) for name in ("torch", "transformers", "recbole",
                                                     "sentence-transformers", "k-means-constrained")},
    }
    write_json(directory / "summary.json", summary)


def evaluate_run(args, data, selected, metadata, device, output):
    from generative import test_checkpoint

    users, items, manifest = data
    directory = args.run_dir
    summary = read_json(directory / "summary.json")
    if metadata["scope"] == "formal":
        require_training_selection(summary)
        require_full_validation(summary.get("validation_protocol"), users)
    elif metadata["scope"] == "pilot":
        require_training_selection(summary)
        if summary.get("validation_protocol") != validation_record(validation_users(users, pilot=True)):
            raise ValueError("LETTER-TIGER checkpoint must use the shared pilot validation cohort")
    sid = "sids.json"
    weights = f"{summary['winner']['trial']}/best.pth"
    if (summary["model"] != MODEL or summary["identity"] != manifest["identity"]
            or summary["evaluation"]["scope"] != metadata["scope"]
            or any(sha256(directory / name) != summary["artifacts"][name] for name in (sid, weights))):
        raise ValueError("LETTER-TIGER checkpoint, data, SIDs or cohort do not match the saved run")
    return test_checkpoint(args, directory, load_catalog(directory / sid, items, args.dataset),
                           selected, metadata, summary, device, output)


def main():
    args = parse_args()
    args.gpu_id = choose_gpu(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    import torch

    device = torch.device("cuda:0" if args.gpu_id else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Pass --device cpu or install CUDA-enabled PyTorch.")
    torch.set_num_threads(4)
    args.run_dir = args.run_dir.resolve()
    data = load_prepared(args.data_dir, args.dataset)
    selected, metadata = load_cohort(args.cohort, args.dataset, data[0], data[1], MAX_HISTORY)
    output = args.output or args.run_dir / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)

    if not args.evaluate_only:
        if args.smoke != (metadata["scope"] == "smoke"):
            raise ValueError("Match --smoke to the scope of the evaluation cohort")
        with staged_output(args.run_dir) as stage:
            train_run(args, stage, data, metadata, device)
    result = evaluate_run(args, data, selected, metadata, device, output)
    print(f"{MODEL} {args.dataset}: saved {result['users']} user rankings to {output}", flush=True)


if __name__ == "__main__":
    main()
