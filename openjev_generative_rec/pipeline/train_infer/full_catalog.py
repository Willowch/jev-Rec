"""Fit an independent full-catalog OpenJev head and evaluate the fixed cohort once."""

import argparse
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch.nn import functional as F

from ..common import enable_comparison, sha256
from ..data_preparation.build_jev_data import SequenceData, input_record
from ..data_preparation.embedding_cache import encoder_fingerprint
from ..data_preparation.item_embeddings import load_encoder, mean_pool
from ..models.full_catalog import FullCatalogHead, catalog_scores, standardize, target_ranks
from ..models.openjev_runtime import load_model
from .gspo_selection import checkpoint_digest, require_sft_source
from .recommend import checkpoint_config
from .test_shards import merge_shards, read_shard, shard_specs

enable_comparison()
from comparison.prepare import load_metadata, load_prepared
from comparison.protocol import (COUNTS, MAX_HISTORY, SEED, TOP_K, TRAIN_SELECTION, VALIDATION_K,
                                 checked_ranks, holdout, load_cohort, ranked_entry, ranking_metrics,
                                 item_key, read_evaluation, read_json, read_rows, sampled_users, staged_output,
                                 train_position, validation_record, write_json)


METHOD = "openjev_full_catalog_v1"
FORMAL_DATASETS = ("ml1m", "Industrial_and_Scientific")
QUERY = "Predict the next item from the interaction history."
ITEM_TOKEN_LIMIT = 128
FEATURE_BLOCK = 512
SEMANTIC_WEIGHTS = (0.125, 0.25, 0.5, 1.0)
POPULARITY_WEIGHTS = (0.0, 0.05, 0.15)
WEIGHTS = [(semantic, popular) for semantic in SEMANTIC_WEIGHTS for popular in POPULARITY_WEIGHTS]


def log(event, **details):
    print(json.dumps({"event": event, **details}, allow_nan=False), flush=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_vectors(path, key, count, build):
    if not path.exists():
        vectors = build().cpu().float().contiguous()
        check_vectors(vectors, count)
        save_tensor(path, {"key": key, "vectors": vectors})
        return vectors
    cached = torch.load(path, weights_only=True, map_location="cpu")
    if cached["key"] != key:
        raise ValueError(f"Feature inputs differ from the existing cache: {path}")
    return check_vectors(cached["vectors"], count)


def check_vectors(vectors, count):
    if vectors.ndim != 2 or vectors.shape[0] != count or not torch.isfinite(vectors).all():
        raise ValueError(f"Expected {count} rows of finite feature vectors")
    return vectors


def split_user(user, split):
    if split == "train":
        position = train_position(user)
        return user["items"][:position], user["items"][position]
    return holdout(user, split)


def queries_for(model, histories, item_codes):
    for history in histories:
        # HistoryMemory verifies that text, SID and item ID stay aligned.
        recent = history[-MAX_HISTORY:]
        row = input_record([item_codes[item] for item in recent], (), recent)
        state = model.prepare_record(row)["state"]
        text = f"History (oldest to newest): {state['history']}\n{state['history_text_memory']}\n{QUERY}"
        yield model.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )


class FeatureBatches:
    """Keep the reduced batch size after an OOM and save only complete blocks."""

    def __init__(self, size, device):
        self.size = size
        self.device = device

    def encode(self, rows, encode_batch):
        start = 0
        parts = []
        while start < len(rows):
            stop = min(start + self.size, len(rows))
            try:
                vectors = encode_batch(rows[start:stop])
            except torch.OutOfMemoryError:
                if self.device.type != "cuda" or self.size == 1:
                    raise
            else:
                parts.append(vectors)
                start = stop
                continue
            self.size = max(1, self.size // 2)
            torch.cuda.empty_cache()
            log("reduced_feature_batch", batch_size=self.size)
        return torch.cat(parts)


class HistoryEncoder:
    def __init__(self, args, data, device):
        self.args, self.data, self.device = args, data, device
        self.batches = FeatureBatches(args.feature_batch, device)
        self.model = None

    @torch.inference_mode()
    def encode(self, histories):
        if self.model is None:
            self.model = load_model(
                self.args.sft_checkpoint, self.args.base_model, device=str(self.device),
                item_catalog=self.args.sid_data / "item_semantic_ids.jsonl",
            ).eval()
            if getattr(self.model, "history_memory", None) is None:
                raise ValueError("Use a completed layered SFT checkpoint with text history memory")
        model = self.model

        def forward(prompts):
            inputs = model.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[1] > model.max_length:
                raise ValueError("A full-catalog history exceeds the source checkpoint context limit")
            inputs = inputs.to(self.device)
            hidden = model.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            last = inputs["attention_mask"].sum(dim=1) - 1
            return hidden[torch.arange(len(last), device=self.device), last].float().cpu()

        return self.batches.encode(list(queries_for(model, histories, self.data.item_ids)), forward)

    def close(self):
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def history_block(args, identity, split, users, start, stop, encoder):
    path = args.output / "features" / split / f"{start:05d}-{stop:05d}.pt"
    key = {"inputs": identity, "split": split, "range": [start, stop],
           "users": fingerprint([user["user_id"] for user in users[start:stop]])}
    started = perf_counter()
    exists = path.is_file()
    vectors = cached_vectors(path, key, stop - start, lambda: encoder.encode(
        [split_user(user, split)[0] for user in users[start:stop]],
    ))
    if not exists:
        elapsed = perf_counter() - started
        log("feature_block", split=split, users_done=stop, users_total=len(users),
            seconds=elapsed, seconds_per_user=elapsed / (stop - start), batch_size=encoder.batches.size)
    return vectors


def history_vectors(args, identity, split, users, encoder):
    # Publish and time the first 128 train histories before extracting the rest.
    first = min(len(users), 128 if split == "train" else FEATURE_BLOCK)
    starts = [0, *range(first, len(users), FEATURE_BLOCK), len(users)]
    return torch.cat([history_block(args, identity, split, users, start, stop, encoder)
                      for start, stop in zip(starts[:-1], starts[1:], strict=True)])


def import_item_features(directory, catalog, identity, encoder_hash):
    expected = {"dataset": identity["dataset"], "catalog_sha256": identity["catalog_sha256"],
                "items_sha256": fingerprint(catalog), "encoder_sha256": encoder_hash,
                "max_length": ITEM_TOKEN_LIMIT, "encoding": "attention-masked-mean-l2-v1"}
    if not directory.is_dir():
        return None
    try:
        record = read_json(directory / "manifest.json")
        if any(record.get("cache_inputs", {}).get(key) != value for key, value in expected.items()):
            return None
        for name in ("items.jsonl", "embeddings.npy"):
            if sha256(directory / name) != record.get("cache_files", {}).get(name):
                return None
        if read_rows(directory / "items.jsonl") != catalog:
            return None
        return check_vectors(torch.from_numpy(np.load(directory / "embeddings.npy", allow_pickle=False)),
                             len(catalog))
    except (OSError, ValueError, TypeError):
        return None


def item_feature_catalog(dataset, base_model, items, metadata):
    if dataset != "Industrial_and_Scientific":
        return metadata

    from transformers import AutoTokenizer

    from ..data_preparation.prepare_amazon import build_catalog, openjev_metadata

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    catalog, _ = build_catalog(dataset, items, openjev_metadata(metadata), tokenizer)
    return catalog


def make_item_features(args, catalog, device):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, _, _ = load_encoder(str(args.base_model), device.type)
    batches = FeatureBatches(args.feature_batch, device)

    @torch.inference_mode()
    def forward(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=ITEM_TOKEN_LIMIT,
                           return_tensors="pt").to(device)
        output = model(**inputs, use_cache=False, return_dict=True)
        return F.normalize(mean_pool(output.last_hidden_state, inputs["attention_mask"]), dim=-1).cpu()

    return batches.encode([row["text"] for row in catalog], forward)


def item_vectors(args, catalog, identity, base_hash, device):
    path = args.output / "features" / "items.pt"
    source = args.item_embeddings or args.sid_data.parent / "embeddings"
    key = {"base_encoder": base_hash, "catalog": fingerprint(catalog), "max_length": ITEM_TOKEN_LIMIT}

    def build():
        vectors = import_item_features(source, catalog, identity, base_hash)
        if vectors is not None:
            log("item_features_reused", path=str(source), items=len(catalog))
            return vectors
        log("item_features_encoding", items=len(catalog))
        return make_item_features(args, catalog, device)

    return cached_vectors(path, key, len(catalog), build)


def ranking_inputs(users, split, item_index, *, add_seen=False):
    history = torch.full((len(users), MAX_HISTORY), len(item_index), dtype=torch.long)
    targets = torch.empty(len(users), dtype=torch.long)
    seen = torch.zeros((len(users), len(item_index)), dtype=torch.bool) if add_seen else None
    for index, user in enumerate(users):
        full_history, target = split_user(user, split)
        recent = [item_index[item] for item in full_history[-MAX_HISTORY:]]
        history[index, :len(recent)] = torch.tensor(recent, dtype=torch.long)
        targets[index] = item_index[target]
        if seen is not None:
            seen[index, [item_index[item] for item in set(full_history)]] = True
    return history, targets, seen


def train_epoch(model, optimizer, history, contexts, targets, batch_size, epoch):
    model.train()
    torch.manual_seed(SEED + epoch)
    order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(SEED + epoch))
    total = targets.new_zeros((), dtype=torch.float32)
    for rows in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        collaborative, semantic = model(history[rows], contexts[rows])
        labels = targets[rows]
        loss = F.cross_entropy(collaborative + semantic, labels)
        loss = loss + 0.2 * (F.cross_entropy(collaborative, labels) + F.cross_entropy(semantic, labels))
        loss.backward()
        optimizer.step()
        total += loss.detach() * len(rows)
    average = float(total) / len(targets)
    if not math.isfinite(average):
        raise FloatingPointError("The full-catalog head produced a non-finite loss")
    return average


@torch.inference_mode()
def validate(model, history, contexts, targets, seen, popularity, batch_size):
    model.eval()
    measures = torch.zeros(len(WEIGHTS) + 1, dtype=torch.float64, device=targets.device)
    for start in range(0, len(targets), batch_size):
        rows = slice(start, start + batch_size)
        collaborative, semantic = (standardize(score) for score in model(history[rows], contexts[rows]))
        for index, (semantic_weight, popular_weight) in enumerate([*WEIGHTS, (0.0, 0.0)]):
            scores = catalog_scores(collaborative, semantic, popularity, semantic_weight, popular_weight)
            ranks = target_ranks(scores, targets[rows], seen[rows])
            measures[index] += ((ranks <= VALIDATION_K) / torch.log2(ranks.double() + 1)).sum()
    values = (measures / len(targets)).tolist()
    candidates = [{"semantic_weight": sem, "popularity_weight": pop, "ndcg_at10": value}
                  for (sem, pop), value in zip(WEIGHTS, values[:-1], strict=True)]
    return candidates, values[-1]


def fit_head(args, run, items, train_users, valid_users, item_features, encoder, identity, device):
    selected = args.output / "selected"
    if selected.exists():
        return
    train_ctx = history_vectors(args, identity, "train", train_users, encoder)
    valid_ctx = history_vectors(args, identity, "validation", valid_users, encoder)
    encoder.close()
    features = item_features()  # The fallback text encoder never shares GPU memory with the SFT encoder.
    if train_ctx.shape[1] != features.shape[1] or valid_ctx.shape[1] != features.shape[1]:
        raise ValueError("OpenJev history and item embeddings have incompatible dimensions")
    index = {item: row for row, item in enumerate(items)}
    train_hist, train_targets, _ = ranking_inputs(train_users, "train", index)
    valid_hist, valid_targets, seen = ranking_inputs(valid_users, "validation", index, add_seen=True)
    popularity = standardize(torch.log1p(torch.bincount(train_targets, minlength=len(items)).float())).to(device)
    train_hist, train_ctx, train_targets = (x.to(device) for x in (train_hist, train_ctx, train_targets))
    valid_hist, valid_ctx, valid_targets, seen = (x.to(device) for x in (valid_hist, valid_ctx, valid_targets, seen))
    torch.manual_seed(SEED)
    model = FullCatalogHead(features.to(device), MAX_HISTORY).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    progress_file = args.output / "head-training.pt"
    state = {"epoch": 0, "best": None, "best_weights": None, "misses": 0, "validation": []}
    state_key = fingerprint(run)
    movie_features_hash = sha256(args.output / "features" / "items.pt")
    if progress_file.exists():
        state = torch.load(progress_file, weights_only=True, map_location="cpu")
        if state["key"] != state_key or state["item_features_sha256"] != movie_features_hash:
            raise ValueError("The saved training state belongs to a different experiment")
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log("fit_resumed", epoch=state["epoch"])
    for epoch in range(state["epoch"] + 1, args.epochs + 1):
        if state["misses"] >= 5:
            break
        loss = train_epoch(model, optimizer, train_hist, train_ctx, train_targets, args.train_batch, epoch)
        scheduler.step()
        log("train_epoch", epoch=epoch, ceiling=args.epochs, loss=loss)
        if epoch % 2 == 0 or epoch == args.epochs:
            candidates, ablation = validate(
                model, valid_hist, valid_ctx, valid_targets, seen, popularity, args.train_batch,
            )
            winner = max(candidates, key=lambda entry: entry["ndcg_at10"])
            if state["best"] is None or winner["ndcg_at10"] > state["best"]["ndcg_at10"]:
                state.update(best={"epoch": epoch, **winner},
                             best_weights={key: val.detach().cpu().clone() for key, val in model.state_dict().items()},
                             misses=0)
            elif epoch > 10:
                state["misses"] += 1
            state["validation"].append({"epoch": epoch, "collaborative_only_ndcg_at10": ablation,
                                        "candidates": candidates})
            log("validation", epoch=epoch, users=len(valid_users), collaborative_only_ndcg_at10=ablation,
                best=state["best"], misses=state["misses"])
        state.update(key=state_key, epoch=epoch, item_features_sha256=movie_features_hash,
                     weights=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict())
        save_tensor(progress_file, state)
    with staged_output(selected) as stage:
        weights = stage / "head.pt"
        torch.save({"weights": state["best_weights"], "popularity": popularity.cpu()}, weights)
        selection = {"run_sha256": state_key, "train_selection": TRAIN_SELECTION,
                     "item_features_sha256": movie_features_hash,
                     "validation_protocol": validation_record(valid_users, smoke=args.smoke),
                     "best_validation": {**state["best"], "checkpoint_sha256": sha256(weights)},
                     "epochs_trained": state["epoch"], "train_users": len(train_users)}
        write_json(stage / "selection.json", selection)
        write_json(stage / "validation.json", state["validation"])
    log("checkpoint_selected", **selection["best_validation"])


@torch.inference_mode()
def test_blocks(args, items, users, item_features, encoder, identity, specs, selection, device):
    weights_path = args.output / "selected" / "head.pt"
    model = None
    index = {item: row for row, item in enumerate(items)}
    chosen = selection["best_validation"]
    for path, meta in specs:
        start, stop = meta["start"], meta["stop"]
        shard_users = users[start:stop]
        if path.exists():
            checked_ranks(read_shard(path, meta), shard_users, items)
            continue
        contexts = history_block(args, identity, "test", users, start, stop, encoder).to(device)
        if model is None:
            features = item_features().to(device)
            model = FullCatalogHead(features, MAX_HISTORY).to(device).eval()
            state = torch.load(weights_path, weights_only=True, map_location=device)
            model.load_state_dict(state["weights"])
            popularity = state["popularity"]
        histories, targets, seen = ranking_inputs(shard_users, "test", index, add_seen=True)
        entries = []
        for offset in range(0, len(shard_users), args.train_batch):
            rows = slice(offset, offset + args.train_batch)
            collaborative, semantic = (standardize(v) for v in model(histories[rows].to(device), contexts[rows]))
            scores = catalog_scores(collaborative, semantic, popularity,
                                    chosen["semantic_weight"], chosen["popularity_weight"])
            scores = scores.masked_fill(seen[rows].to(device), -torch.inf)
            order = torch.argsort(scores, descending=True, stable=True)[:, :TOP_K]
            for user, target, ids, values in zip(shard_users[rows], targets[rows].tolist(), order.cpu().tolist(),
                                                scores.gather(1, order).cpu().tolist(), strict=True):
                recommendations = [{"item_id": items[item], "score": score}
                                   for item, score in zip(ids, values, strict=True) if math.isfinite(score)]
                entries.append(ranked_entry(user["user_id"], items[target], recommendations))
        checked_ranks(entries, shard_users, items)
        with staged_output(path) as stage:
            write_json(stage / "rankings.json", {"metadata": meta, "entries": entries})
        log("test_block", users_done=stop, users_total=len(users), block=path.name)
    encoder.close()


def compare_baselines(args, final, items, users):
    from comparison.summarize import summarize

    table = summarize([final, *args.baseline_results], args.comparison_data, args.cohort, args.dataset, True)
    baseline_scores = {}
    for path in args.baseline_results:
        result, ranks = read_evaluation(path, users, items)
        baseline_scores[result["model"]] = ranking_metrics(ranks, (VALIDATION_K,))["NDCG@10"]
    _, ranks = read_evaluation(final, users, items)
    score = ranking_metrics(ranks, (VALIDATION_K,))["NDCG@10"]
    leader = max(baseline_scores, key=baseline_scores.get)
    report = {"openjev_ndcg_at10": score, "best_baseline": leader, "baseline_ndcg_at10": baseline_scores[leader],
              "delta_ndcg_at10": score - baseline_scores[leader], "exceeds_baseline": score > baseline_scores[leader],
              "baselines_ndcg_at10": baseline_scores}
    summary = args.output / "comparison.json"
    if summary.exists():
        if read_json(summary) != report:
            raise ValueError("The supplied baseline results differ from the recorded comparison")
    else:
        write_json(summary, report)
    (args.output / "comparison.md").write_text(table, encoding="utf-8")
    print(table, end="")
    log("baseline_comparison", **report)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "fit", "test", "merge"))
    for name in ("comparison-data", "cohort", "sid-data", "sft-checkpoint", "base-model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=(*FORMAL_DATASETS, "ml100k"), default="ml1m")
    parser.add_argument("--item-embeddings", type=Path)
    parser.add_argument("--baseline-results", type=Path, nargs=4)
    parser.add_argument("--feature-batch", type=int, default=32)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Use 16 users per split for a local pipeline check")
    args = parser.parse_args()
    if min(args.feature_batch, args.train_batch, args.epochs) < 1:
        parser.error("Batch sizes and epoch count must be positive")
    if not args.smoke and args.dataset not in FORMAL_DATASETS:
        parser.error("ML-100K can only be used with --smoke")
    if args.baseline_results and (args.smoke or not all(path.is_file() for path in args.baseline_results)):
        parser.error("--baseline-results requires the four existing formal evaluation files")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.baseline_results:
        args.baseline_results = [path.resolve() for path in args.baseline_results]
    return args


def execute(args):
    device = torch.device(args.device)
    if args.stage != "merge" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    if device.type == "cuda" and device.index not in (None, 0):
        raise ValueError("Expose the assigned GPU as cuda:0")
    users, items, prepared = load_prepared(args.comparison_data, args.dataset)
    if len(items) != len(set(items)):
        raise ValueError("The prepared item catalog contains duplicate IDs")
    items = sorted(items, key=lambda item: item_key(args.dataset, item))
    metadata = load_metadata(args.comparison_data, items, prepared)
    data = SequenceData(args.sid_data, args.comparison_data)
    require_sft_source(args.sft_checkpoint, data)
    config = checkpoint_config(args.sft_checkpoint, args.sid_data / "item_semantic_ids.jsonl")
    if "history_memory" not in read_json(args.sft_checkpoint / "model.json") or config["max_history"] != MAX_HISTORY:
        raise ValueError("A completed layered SFT checkpoint with 20-item text history is required")
    selected, cohort = load_cohort(args.cohort, args.dataset, users, items, MAX_HISTORY)
    if not args.smoke:
        expected_users, expected_items, expected_test = COUNTS[args.dataset]
        if (cohort["scope"] != "formal" or (len(users), len(items), len(selected)) !=
                (expected_users, expected_items, expected_test)):
            raise ValueError(f"Formal {args.dataset} scoring needs {expected_users:,} training and validation users, "
                             f"{expected_items:,} items and {expected_test:,} test users")
    train_users, valid_users, test_users = users, users, selected
    if args.smoke:
        train_users, valid_users, test_users = (sampled_users(rows, 16) for rows in (users, users, selected))
        cohort = {**cohort, "scope": "smoke",
                  "cohort_sha256": fingerprint([row["user_id"] for row in test_users])}
    base_hash = encoder_fingerprint(args.base_model)
    identity = {"method": METHOD, "dataset": prepared["identity"], "metadata": prepared["metadata_sha256"],
                "sid": config["semantic_ids_sha256"], "source_checkpoint": checkpoint_digest(args.sft_checkpoint),
                "base_encoder": base_hash, "history_query": QUERY,
                "runtime": {name: version(name) for name in ("torch", "transformers", "peft")},
                "encoder_options": {name: os.environ[name] for name in (
                    "JEV_LOAD_4BIT", "JEV_LOAD_8BIT", "JEV_TORCH_DTYPE", "OPENJEV_REFERENCE_KERNELS",
                    "OPENJEV_COMPACT_DELTA_RULE", "OPENJEV_CPU_TOKEN_EMBEDDINGS", "JEV_DEVICE_MAP",
                ) if name in os.environ}}
    run = {"features": identity, "cohort": cohort, "epochs": args.epochs, "train_batch": args.train_batch,
           "seed": SEED, "validation_protocol": validation_record(valid_users, smoke=args.smoke)}
    run_config = args.output / "run.json"
    if run_config.exists():
        if read_json(run_config) != run:
            raise ValueError("Output directory has different training or feature inputs; choose a new one")
    else:
        write_json(run_config, run)
    encoder = HistoryEncoder(args, data, device)

    def load_items():
        catalog = item_feature_catalog(args.dataset, args.base_model, items, metadata)
        return item_vectors(args, catalog, prepared["identity"], base_hash, device)

    if args.stage in ("run", "fit"):
        fit_head(args, run, items, train_users, valid_users, load_items, encoder, identity, device)
    if args.stage == "fit":
        return
    selection = read_json(args.output / "selected" / "selection.json")
    weights = args.output / "selected" / "head.pt"
    if (selection["run_sha256"] != fingerprint(run)
            or selection["best_validation"]["checkpoint_sha256"] != sha256(weights)
            or selection["item_features_sha256"] != sha256(args.output / "features" / "items.pt")
            or selection["validation_protocol"] != run["validation_protocol"]):
        raise ValueError("The selected head or its validation record does not match this run")
    details = {"ranking_mode": "full_catalog", "method": METHOD, "seed": SEED, "catalog_size": len(items),
               "source_sft_sha256": identity["source_checkpoint"],
               "checkpoint_sha256": selection["best_validation"]["checkpoint_sha256"],
               "protocol": "full catalog, seen items excluded, exact item scoring"}
    specs = shard_specs(args.output / "test-shards", len(test_users), FEATURE_BLOCK, {**cohort, **details})
    final = args.output / "evaluation.json"
    if final.exists():
        result, _ = read_evaluation(final, test_users, items)
        if any(result.get(key) != value for key, value in {**cohort, **details}.items()):
            raise ValueError("The existing test result belongs to a different model or cohort")
        log("evaluation_reused", output=str(final))
    else:
        if args.stage in ("run", "test"):
            test_blocks(args, items, test_users, load_items, encoder, identity, specs, selection, device)
        result = merge_shards(final, specs, test_users, items, cohort, details, selection)
        log("evaluation_complete", output=str(final), **result)
    if args.baseline_results:
        compare_baselines(args, final, items, test_users)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(args.output / ".run.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError(f"Another full-catalog process is already using {args.output}") from error
    try:
        execute(args)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
