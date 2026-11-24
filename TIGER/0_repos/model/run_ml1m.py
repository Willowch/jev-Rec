"""Run native TIGER on the same MovieLens-1M protocol as OpenJev SFT."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

from dataloader import GenRecDataLoader
from tiger_model import TIGER, TIGER_MODEL_DEFAULTS, set_seed, train
from amazon_eval import Catalog, recommendation_entries
from experiment_tracking import PROJECT_ROOT, sha256_file, write_json
from ml1m_data import (CODEBOOK_SIZE, SID_LAYERS, SOURCE_LAYERS, TrainingPairs,
                       history_tokens, holdout, load_comparison_sequences, load_native_codes, load_sequences,
                       model_tokens, token_offset)
from ml1m_eval import PROTOCOL, comparison_protocol, evaluate_cases, fit_temperature, recommend, select_users


DEFAULT_SENTENCE_MODEL = (PROJECT_ROOT / "02_data/3_models/huggingface"
                          / "models--sentence-transformers--sentence-t5-base/snapshots"
                          / "fc5d4628481afbbaaacd7af6bb07cf9d3865f781")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openjev-root", type=Path,
                        default=PROJECT_ROOT.parent / "openjev_generative_rec")
    parser.add_argument("--data", type=Path, help="Legacy OpenJev SequenceData directory")
    parser.add_argument("--dataset", choices=("ml1m", "ml100k"), default="ml1m")
    parser.add_argument("--comparison-data", type=Path, help="Shared sequence and product metadata directory")
    parser.add_argument("--sentence-model", type=Path, default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument("--run-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--evaluate-only", action="store_true", help="Reload a trained run for test evaluation")
    parser.add_argument("--output", type=Path, help="New output JSON for --evaluate-only")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Use the Qwen SFT smoke protocol")
    parser.add_argument("--rqvae-epochs", type=int)
    parser.add_argument("--tiger-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-steps", type=int, help="Optimizer step cap per TIGER epoch")
    parser.add_argument("--eval-step", type=int, default=5)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-rows", type=int)
    parser.add_argument("--validation-users", type=int, help="Standalone validation limit (0 means all)")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    cohort_group = parser.add_mutually_exclusive_group()
    cohort_group.add_argument("--test-users", type=int, help="0 means all test users")
    cohort_group.add_argument("--cohort", type=Path, help="Fixed comparison test cohort")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()
    if args.evaluate_only and args.output is None:
        parser.error("--evaluate-only requires a new --output path")
    if args.dataset == "ml100k" and (not args.comparison_data or not args.cohort or args.data or args.smoke):
        parser.error("The MovieLens-100K pilot needs --comparison-data and --cohort")
    if args.cohort and any(value not in (None, 20) for value in (args.beam_size, args.top_k)):
        parser.error("--cohort requires test evaluation with beam=20 and top-k=20")

    defaults = dict(rqvae_epochs=3000, tiger_epochs=200, batch_size=256, calibration_rows=30,
                    test_users=0, beam_size=10, top_k=10)
    if args.smoke:
        defaults.update(rqvae_epochs=1, tiger_epochs=1, batch_size=16, max_train_steps=3,
                        calibration_rows=3, test_users=1, beam_size=3, top_k=3)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.openjev_root = args.openjev_root.resolve()
    if args.comparison_data is not None:
        args.comparison_data = args.comparison_data.resolve()
    args.run_dir = args.run_dir.resolve()
    args.sentence_model = args.sentence_model.resolve()
    if min(args.rqvae_epochs, args.tiger_epochs, args.batch_size, args.eval_step, args.early_stop,
           args.calibration_rows, args.beam_size, args.top_k, args.eval_batch_size) < 1:
        parser.error("epochs, batch size, evaluation cadence, patience, calibration rows, beam and K must be positive")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be positive")
    if min(args.validation_users or 0, args.test_users) < 0:
        parser.error("user counts must be nonnegative")
    return args


def embed_catalog(catalog, model_path, device, output):
    comparison_protocol()
    from comparison.features import encode_sentence_t5

    encode_sentence_t5(catalog, "ml1m", model_path, device, 32, output)


def run_rqvae_script(script, options, log):
    command = [sys.executable, "-B", "-u", str(PROJECT_ROOT / "0_repos/rqvae" / script), *options]
    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "2")
    print(f"Running RQ-VAE {script}. Log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"RQ-VAE {script} exited with {result.returncode}. See {log}")


def generate_native_codes(args, embedding_path, code_path):
    run_root = args.run_dir / "rqvae"
    dataset = "ML-100K" if args.dataset == "ml100k" else "ML-1M"
    train_options = ["--data_path", str(embedding_path), "--dataset", dataset, "--disable_mlflow",
                     "--run_root", str(run_root), "--device", args.device, "--num_workers", "0",
                     "--seed", str(args.seed), "--epochs", str(args.rqvae_epochs)]
    if args.smoke or args.dataset == "ml100k":
        train_options += ["--warmup_epochs", "0"]
    run_rqvae_script("main.py", train_options, args.run_dir / "rqvae_train.log")
    (summary_path,) = run_root.glob(f"{dataset}/stage1_rqvae/*/summary.json")
    checkpoint = json.loads(summary_path.read_text(encoding="utf-8"))["checkpoint_path"]
    export_options = ["--ckpt_path", checkpoint, "--data_path", str(embedding_path),
                      "--output_file", str(code_path), "--device", args.device, "--disable_mlflow"]
    run_rqvae_script("generate_code.py", export_options, args.run_dir / "rqvae_export.log")


class HistoryScorer:
    """Encode a history once and score each layer for a batch of generated prefixes."""

    def __init__(self, model, history, item_tokens, max_history, device):
        self.model = model.model
        self.device = device
        encoded = history_tokens(history, item_tokens, max_history)
        ids = torch.tensor([encoded], device=device).flatten(1)
        self.mask = ids.ne(0).long()
        self.hidden = self.model.get_encoder()(input_ids=ids, attention_mask=self.mask).last_hidden_state

    def __call__(self, prefixes):
        layer = len(prefixes[0])
        decoder_ids = torch.tensor([(0,) + model_tokens(prefix) for prefix in prefixes],
                                   device=self.device)
        outputs = self.model(encoder_outputs=(self.hidden.expand(len(prefixes), -1, -1),),
                             attention_mask=self.mask.expand(len(prefixes), -1),
                             decoder_input_ids=decoder_ids, use_cache=False)
        start = token_offset(layer)
        return outputs.logits[:, -1, start:start + CODEBOOK_SIZE]


@torch.no_grad()
def calibrate(model, data, item_codes, item_tokens, args, device):
    # Match train_lora.py's sampling of OpenJev's three decision rows per user.
    count = data.count("calibration")
    indices = random.Random(args.seed + 1).sample(range(count), min(args.calibration_rows, count))
    logits, labels = [], []
    for index in indices:
        user_index, layer = divmod(index, SOURCE_LAYERS)
        history, target = holdout(data.users[user_index], "calibration")
        code = item_codes[target]
        scorer = HistoryScorer(model, history, item_tokens, data.manifest["max_history"], device)
        logits.append(scorer([code[:layer]])[0].float().cpu().tolist())
        labels.append(code[layer])
    return {"temperature": fit_temperature(logits, labels), "split": "calibration",
            "indices": indices, "normalization": "full"}


@torch.no_grad()
def evaluate_split(model, users, split, item_codes, item_tokens, max_history, temperature, args, device):
    def predict(history):
        scorer = HistoryScorer(model, history, item_tokens, max_history, device)
        return recommend(item_codes, history, scorer, layers=SID_LAYERS, beam_size=args.beam_size,
                         top_k=args.top_k, temperature=temperature)

    cases = ((user["user_id"], *holdout(user, split)) for user in users)
    return evaluate_cases(cases, predict, beam_size=args.beam_size, top_k=args.top_k, seed=args.seed)


def train_tiger(data, item_codes, args, device, *, shared=False):
    set_seed(args.seed)
    model = TIGER(TIGER_MODEL_DEFAULTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    item_tokens = {item: model_tokens(code) for item, code in item_codes.items()}
    train_loader = GenRecDataLoader(TrainingPairs(data, item_tokens), batch_size=args.batch_size,
                                    num_workers=0)
    max_history = data.manifest["max_history"]
    selection = None
    if shared:
        if args.validation_users is not None:
            raise ValueError("Shared-data training selects its own fixed validation users")
        protocol = comparison_protocol()
        validation_users = protocol.validation_users(data.users, smoke=args.smoke, pilot=args.dataset == "ml100k")
        selection = protocol.validation_record(validation_users, smoke=args.smoke)
        catalog = Catalog(item_codes, args.dataset)
        cutoff = protocol.VALIDATION_K
    else:
        limit = args.validation_users if args.validation_users is not None else int(args.smoke)
        validation_users = select_users(data.users, args.seed, limit)
        cutoff = args.top_k
    checkpoint = args.run_dir / "checkpoint"
    checkpoint.mkdir()
    weights_path = checkpoint / "best_model.pth"
    calibration_path = checkpoint / "temperature.json"
    best_score = -math.inf
    optimizer_steps = 0
    metrics = []
    for epoch in range(1, args.tiger_epochs + 1):
        loss = train(model, train_loader, optimizer, device, max_steps=args.max_train_steps)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite TIGER loss at epoch {epoch}")
        optimizer_steps += min(len(train_loader), args.max_train_steps or len(train_loader))
        epoch_metrics = {"epoch": epoch, "loss": loss, "optimizer_steps": optimizer_steps}
        metrics.append(epoch_metrics)
        print(f"TIGER epoch {epoch}: loss={loss:.6f}, updates={optimizer_steps}", flush=True)
        if epoch % args.eval_step and epoch != args.tiger_epochs:
            continue

        model.eval()
        calibration = calibrate(model, data, item_codes, item_tokens, args, device)
        if shared:
            entries = recommendation_entries(
                model, catalog, validation_users, "validation", calibration["temperature"], device,
                max_history=max_history, batch_size=args.eval_batch_size, beam_size=cutoff, top_k=cutoff)
            validation = {"beam_size": cutoff,
                          "protocol": PROTOCOL, "policy_normalization": "full", "seed": args.seed,
                          **protocol.validation_metrics(entries, validation_users)}
        else:
            validation = evaluate_split(model, validation_users, "validation", item_codes, item_tokens,
                                        max_history, calibration["temperature"], args, device)
        score = validation[f"NDCG@{cutoff}"]
        epoch_metrics.update(validation_ndcg=score)
        print(f"Validation NDCG@{cutoff}={score:.6f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), weights_path)
            write_json(calibration_path, calibration)
            write_json(args.run_dir / "validation.json", validation)
        elif epoch - best_epoch >= args.early_stop:
            break

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    return {"best_epoch": best_epoch, "epochs_trained": epoch, "training": metrics,
            "optimizer_steps": optimizer_steps, "calibration": calibration, "validation_protocol": selection}


def evaluate_checkpoint(data, item_codes, args, device, cohort, selection, output):
    checkpoint = args.run_dir / "checkpoint"
    model = TIGER(TIGER_MODEL_DEFAULTS).to(device)
    model.load_state_dict(torch.load(checkpoint / "best_model.pth", map_location=device))
    model.eval()
    calibration = json.loads((checkpoint / "temperature.json").read_text(encoding="utf-8"))
    if cohort:
        users, metadata = cohort
        cutoff = metadata["top_k"]
        entries = recommendation_entries(
            model, Catalog(item_codes, args.dataset), users, "test", calibration["temperature"], device,
            max_history=data.manifest["max_history"], batch_size=args.eval_batch_size, beam_size=cutoff, top_k=cutoff)
        return comparison_protocol().write_evaluation(
            output, entries, users, item_codes, metadata, "TIGER", protocol=PROTOCOL,
            policy_normalization="full", beam_size=cutoff, seed=args.seed, validation_protocol=selection,
            train_selection=comparison_protocol().TRAIN_SELECTION)
    users = select_users(data.users, args.seed, args.test_users)
    item_tokens = {item: model_tokens(code) for item, code in item_codes.items()}
    result = evaluate_split(model, users, "test", item_codes, item_tokens, data.manifest["max_history"],
                            calibration["temperature"], args, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return {key: value for key, value in result.items() if key != "entries"}


def main():
    sys.dont_write_bytecode = True
    args = parse_args()
    if not args.evaluate_only and not (args.sentence_model / "modules.json").is_file():
        raise FileNotFoundError(f"Local Sentence-T5 snapshot is missing: {args.sentence_model}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or a CUDA-enabled TIGER environment.")
    torch.set_num_threads(2)
    summary = (json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
               if args.evaluate_only else None)
    if summary is None:
        canonical = args.data is None and bool(args.comparison_data or args.cohort)
    else:
        canonical = "comparison_identity" in summary
    shared = args.comparison_data
    if canonical:
        if args.data is not None:
            raise ValueError("Use --comparison-data to reload a canonical TIGER run")
        shared = shared or comparison_protocol().DATA_ROOT / args.dataset
        data, catalog, prepared = load_comparison_sequences(shared, args.dataset)
        inputs = {"comparison_identity": prepared["identity"], "metadata_sha256": prepared["metadata_sha256"]}
    else:
        args.data = (args.data or args.openjev_root / "data/ml1m_qwen_sequences").resolve()
        data, catalog = load_sequences(args.openjev_root, args.data, shared)
        hashes = {name: sha256_file(args.data / name) for name in ("manifest.json", "item_semantic_ids.jsonl")}
        hashes["sequences.jsonl"] = sha256_file(data.sequence_path)
        inputs = {"input_sha256": hashes}
        if shared is None and data.sequence_path.parent != args.data:
            shared = data.sequence_path.parent
    item_ids = [row["item_id"] for row in catalog]
    cohort = (comparison_protocol().load_cohort(args.cohort, args.dataset, data.users, item_ids,
                                                data.manifest["max_history"]) if args.cohort else None)
    smoke = args.smoke if summary is None else summary["configuration"]["smoke"]
    if cohort and smoke != (cohort[1]["scope"] == "smoke"):
        raise ValueError("Match the TIGER training scope to the evaluation cohort")
    if not smoke and shared is not None and not cohort:
        raise ValueError("Formal shared-data runs require a full test cohort")
    if summary is None and (canonical or shared is not None or cohort) and args.validation_users is not None:
        raise ValueError("Shared-data training selects its own fixed validation users")
    embedding_path = args.run_dir / "item_emb.parquet"
    code_path = args.run_dir / "item_sid.npy"
    output = args.output or args.run_dir / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    if args.evaluate_only:
        selection = summary.get("validation_protocol")
        if not smoke and (shared is not None or cohort):
            comparison_protocol().require_training_selection(summary)
            if args.dataset == "ml100k":
                protocol = comparison_protocol()
                if selection != protocol.validation_record(protocol.validation_users(data.users, pilot=True)):
                    raise ValueError("TIGER checkpoint must use the shared pilot validation cohort")
            else:
                comparison_protocol().require_full_validation(selection, data.users)
        if reference := summary.get("embedding_ref"):
            embedding_path = (args.run_dir / reference).resolve()
        if (any(summary.get(name) != value for name, value in inputs.items())
                or summary["model"] != TIGER_MODEL_DEFAULTS
                or summary["embedding_sha256"] != sha256_file(embedding_path)
                or summary["native_sid_sha256"] != sha256_file(code_path)):
            raise ValueError("TIGER checkpoint inputs differ from the existing run")
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        if shared is not None:
            comparison_protocol()
            from comparison.features import sentence_t5
            from comparison.prepare import load_prepared

            if canonical:
                items = item_ids
            else:
                _, items, prepared = load_prepared(shared, "ml1m")
            embedding_path, _ = sentence_t5(shared, args.dataset, items, prepared, args.sentence_model, device, 32)
            # Existing RQ-VAE checkpoints associate codes with Parquet row IDs.
            by_id = {row["item_id"]: row for row in catalog}
            catalog = [by_id[item] for item in items]
        else:
            embed_catalog(catalog, args.sentence_model, device, embedding_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        generate_native_codes(args, embedding_path, code_path)
    if args.evaluate_only and summary.get("embedding_ref"):
        catalog = sorted(catalog, key=lambda row: int(row["item_id"]))
    item_codes = load_native_codes(code_path, embedding_path, catalog)
    if not args.evaluate_only:
        training = train_tiger(data, item_codes, args, device, shared=bool(shared is not None or cohort))
        selection = training["validation_protocol"]
        write_json(args.run_dir / "summary.json", {
            "configuration": vars(args), **inputs,
            "embedding_sha256": sha256_file(embedding_path), "native_sid_sha256": sha256_file(code_path),
            "embedding_ref": os.path.relpath(embedding_path, args.run_dir)
            if embedding_path.parent != args.run_dir else None,
            "users": len(data.users), "train_pairs": data.train_pairs,
            "train_selection": comparison_protocol().TRAIN_SELECTION,
            "catalog_size": len(item_codes), "max_history": data.manifest["max_history"],
            "sid_layers": SID_LAYERS, "classes_per_layer": CODEBOOK_SIZE, "model": TIGER_MODEL_DEFAULTS,
            **training,
        })
    result = evaluate_checkpoint(data, item_codes, args, device, cohort, selection, output)
    print(json.dumps(result), flush=True)
    print(f"Evaluation: {output}", flush=True)


if __name__ == "__main__":
    main()
