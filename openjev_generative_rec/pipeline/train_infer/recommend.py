"""Autoregressive SID recommendation with catalog-constrained beam search."""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch

from ..common import enable_comparison, sha256
from ..data_preparation.build_jev_data import CatalogPolicy, SEMANTIC_LAYERS, SequenceData, TOKENS, input_record
from ..models.openjev_runtime import load_model, score


SCORING_BACKENDS = ("batched", "prefix-cache")


def add_scoring_backend(parser):
    parser.add_argument("--scoring-backend", choices=SCORING_BACKENDS, default="batched")


def scoring_settings(backend):
    # Keep existing batched fingerprints valid. Cache tuning knobs can affect numerical parity.
    if backend == "batched":
        return {}
    settings = {"scoring_backend": backend, "prefix_cache_options": {
        name: os.environ.get(name) or "0"
        for name in ("JEV_PREFILL_CHUNK", "JEV_NO_QPREFILL", "JEV_RAGGED_SUFFIX", "OPENJEV_REFERENCE_KERNELS")
    }, "backbone_dtype": os.environ.get("JEV_TORCH_DTYPE") or "bfloat16"}
    for name in ("OPENJEV_COMPACT_DELTA_RULE", "OPENJEV_CPU_TOKEN_EMBEDDINGS", "JEV_BATCH_CHOICES",
                 "JEV_LOAD_4BIT", "JEV_LOAD_8BIT"):
        if os.environ.get(name) == "1":
            settings["prefix_cache_options"][name] = "1"
    return settings


@torch.no_grad()
def score_prefixes(model, rows, candidate_batch):
    from jev.serving import candidate_batches

    # A batch may contain pieces of several prefixes. Reassemble each prefix's
    # raw scores before temperature scaling and normalization.
    logits = [[] for _ in rows]
    for batch in candidate_batches(rows, candidate_batch):
        values = model([part for _, part in batch])
        for (index, _), part_logits in zip(batch, values, strict=True):
            logits[index].append(part_logits.float())
    return [torch.cat(parts) for parts in logits]


def recommendation_sort_key(row):
    item = row["item_id"]
    # ML-1M 按数字大小排序，Amazon 的字母数字 ASIN 按字典序排序。
    item_key = (0, int(item)) if item.isascii() and item.isdecimal() else (1, item)
    return -row["log_probability"], item_key


def recommend(model, item_ids, history, *, max_history=20, beam_size=10, top_k=10,
              candidate_batch=1, temperature=1.0, normalization="full", batch_prefixes=False,
              catalog_key=None, catalog_policy=None, scoring_backend="batched", score_trace=None,
              skip_singletons=False):
    if normalization not in ("full", "catalog"):
        raise ValueError("unknown policy normalization")
    if scoring_backend not in SCORING_BACKENDS:
        raise ValueError("unknown scoring backend")
    catalog_policy = catalog_policy or CatalogPolicy(item_ids)
    policy = catalog_policy if normalization == "catalog" else None
    skip_singletons = skip_singletons and policy is not None and score_trace is None
    depth = catalog_policy.depth
    if depth > SEMANTIC_LAYERS and policy is None:
        raise ValueError("Identity SID decoding requires catalog normalization")
    seen = set(history)
    catalog = [(item, tuple(ids)) for item, ids in item_ids.items() if item not in seen]
    allowed = {level: {ids[:level] for _, ids in catalog} for level in range(1, depth + 1)}
    history_items = history[-max_history:]
    history_ids = [item_ids[item] for item in history_items]
    beams = [((), 0.0)]
    for level in range(depth):
        expansions = []
        rows = []
        if not beams:
            break
        for prefix, _ in beams:
            # 模型只接收自己已生成的 prefix，绝不读取测试 target 的前缀。
            if policy is None:
                row = input_record(history_ids, prefix, history_items)
            else:
                # 在模型打分前筛掉非法候选。与 GSPO rollout 一样，已看过的物品仍参与归一化。
                row = policy.row(history_ids, prefix, history_items)
            rows.append(row)
        scoring_indices = [i for i, row in enumerate(rows) if not skip_singletons or len(row["options"]) != 1]
        scoring_rows = [rows[i] for i in scoring_indices]
        if not scoring_rows:
            level_logits = ()
        elif scoring_backend == "prefix-cache":
            level_logits, _ = model.score_cached(scoring_rows, batch_size=candidate_batch)
        elif batch_prefixes:
            level_logits = score_prefixes(model, scoring_rows, candidate_batch)
        else:
            level_logits = (score(model, row, candidate_batch) for row in scoring_rows)
        logits_by_index = dict(zip(scoring_indices, level_logits, strict=True))
        for i, ((prefix, accumulated), row) in enumerate(zip(beams, rows, strict=True)):
            if i not in logits_by_index:
                # A catalog choice with one action has log probability zero.
                extended = prefix + (row["options"][0],)
                if extended in allowed[level + 1]:
                    expansions.append((extended, accumulated))
                continue
            raw_logits = logits_by_index[i]
            if score_trace is not None:
                score_trace(prefix, row["options"], raw_logits.detach().float().cpu())
            logits = raw_logits / temperature
            options = row["options"]
            if policy is not None and depth == SEMANTIC_LAYERS:
                # 恢复原来的 255 位掩码布局，保留 log_softmax 与 beam 排序逻辑。
                indices = [TOKENS.index(token) for token in row["options"]]
                masked = logits.new_full((len(TOKENS),), float("-inf"))
                masked[indices] = logits
                logits = masked
                options = TOKENS
            log_probs = logits.log_softmax(-1).cpu().tolist()
            for token, value in zip(options, log_probs, strict=True):
                extended = prefix + (token,)
                if extended in allowed[level + 1]:
                    expansions.append((extended, accumulated + value))
        beams = sorted(expansions, key=lambda pair: (-pair[1], pair[0]))[:beam_size]
    scores = dict(beams)
    # RQ 可能碰撞；同码物品同分，按 item_id 确定稳定顺序，不使用测试标签解歧义。
    matches = [{"item_id": item, "semantic_ids": ids, "log_probability": scores[ids]}
               for item, ids in catalog if ids in scores]
    if catalog_key is None:
        return sorted(matches, key=recommendation_sort_key)[:top_k]
    return sorted(matches, key=lambda row: (-row["log_probability"], catalog_key(row["item_id"])))[:top_k]


def comparison_cohort(data, config, checkpoint, cohort, base_model=None, *,
                      scoring_backend="batched", candidate_batch=None):
    enable_comparison()
    from comparison.protocol import SPLIT_PROTOCOL, load_cohort

    if (data.manifest["split_protocol"] != SPLIT_PROTOCOL
            or config["max_history"] != data.manifest["max_history"]):
        raise ValueError("checkpoint history or data split differs from the comparison protocol")
    dataset = data.manifest.get("dataset", "ml1m")
    selected, metadata = load_cohort(cohort, dataset, data.users, data.item_ids, config["max_history"])
    selection = None
    if metadata["scope"] == "pilot":
        from comparison.protocol import read_json, validation_record, validation_users
        from .gspo_selection import checkpoint_digest, require_sft_source

        require_sft_source(checkpoint, data)
        selection = read_json(checkpoint.parent / "run.json")
        expected = validation_record(validation_users(data.users, pilot=True))
        if selection.get("validation_protocol") != expected or not selection.get("best_validation"):
            raise ValueError("Pilot SFT checkpoint must be selected on the shared validation cohort")
        selection["best_validation"]["checkpoint_sha256"] = checkpoint_digest(checkpoint)
    elif metadata["scope"] == "formal":
        from .gspo_selection import base_model_settings, require_selected_checkpoint

        selection = require_selected_checkpoint(checkpoint, data)
        settings = selection["validation_settings"]
        actual = base_model_settings(base_model or settings["base_model"])
        if any(settings.get(key) != value for key, value in actual.items()):
            raise ValueError("Base model differs from validation; use the verified base model or revalidate")
        reference_backend = settings.get("scoring_backend", "batched")
        if "prefix-cache" in (scoring_backend, reference_backend):
            if (scoring_backend != reference_backend or candidate_batch != settings.get("candidate_batch")
                    or any(settings.get(key) != value
                           for key, value in scoring_settings(scoring_backend).items())):
                raise ValueError("Formal cached testing requires the same scorer and batch used for validation")
    return selected, metadata, selection


def checkpoint_config(checkpoint, sid_path):
    config = json.loads((checkpoint / "recommendation.json").read_text(encoding="utf-8"))
    if config["semantic_ids_sha256"] != sha256(sid_path):
        raise ValueError("semantic ID mapping differs from training")
    return config


def predictor(args, data, config, dataset=None):
    catalog_key = None
    if dataset is not None:
        from comparison.protocol import item_key

        def catalog_key(item):
            return item_key(dataset, item)

    temperature = json.loads((args.checkpoint / "temperature.json").read_text(encoding="utf-8"))["temperature"]
    model = load_model(args.checkpoint, args.base_model, item_catalog=args.data / "item_semantic_ids.jsonl")

    def predict(history, *, beam_size=None, top_k=None, scoring_backend=None, score_trace=None):
        return recommend(model, data.item_ids, history, max_history=config["max_history"],
                         beam_size=args.beam_size if beam_size is None else beam_size,
                         top_k=args.top_k if top_k is None else top_k,
                         candidate_batch=args.candidate_batch, temperature=temperature,
                         normalization=config.get("policy_normalization", "full"),
                         batch_prefixes=args.batch_prefixes, catalog_key=catalog_key, catalog_policy=data.catalog,
                         scoring_backend=scoring_backend or getattr(args, "scoring_backend", "batched"),
                         score_trace=score_trace, skip_singletons=dataset == "ml100k")

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", help="迁移目录后可显式指定本地基础权重路径")
    parser.add_argument("--data", type=Path, default=Path("data/ml1m_sequences"))
    parser.add_argument("--comparison-data", type=Path, help="Shared sequence directory")
    parser.add_argument("--history", nargs="+", help="物品 ID；不指定则运行测试用户评估")
    cohort_group = parser.add_mutually_exclusive_group()
    cohort_group.add_argument("--test-users", type=int, help="0 表示全部测试用户")
    cohort_group.add_argument("--cohort", type=Path, help="固定对照测试清单")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-batch", type=int, default=1)
    parser.add_argument("--batch-prefixes", action="store_true", help="合并同一 SID 层内多个 beam 的候选打分")
    add_scoring_backend(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/recommendations.json"))
    args = parser.parse_args()
    default_k = 20 if args.cohort else 10
    if args.beam_size is None:
        args.beam_size = default_k
    if args.top_k is None:
        args.top_k = default_k
    if args.test_users is None:
        args.test_users = 10
    if min(args.beam_size, args.top_k, args.candidate_batch) < 1 or args.test_users < 0:
        raise ValueError("invalid beam/k/batch/test-users")
    if args.cohort and (args.history or args.beam_size != default_k or args.top_k != default_k):
        parser.error("--cohort requires test evaluation with beam=20 and top-k=20")
    data = SequenceData(args.data, args.comparison_data) if args.comparison_data else SequenceData(args.data)
    if args.history:
        unknown = sorted(set(args.history).difference(data.item_ids))
        if unknown:
            parser.error(f"history items missing from the catalog: {', '.join(unknown)}")
    config = checkpoint_config(args.checkpoint, args.data / "item_semantic_ids.jsonl")
    dataset = None
    if args.cohort:
        if args.output.exists():
            raise FileExistsError(args.output)
        selected, comparison, selection = comparison_cohort(
            data, config, args.checkpoint, args.cohort, args.base_model,
            scoring_backend=args.scoring_backend, candidate_batch=args.candidate_batch)
        if selection and not args.base_model:
            args.base_model = (selection["arguments"]["base_model"] if comparison["scope"] == "pilot"
                               else selection["validation_settings"]["base_model"])
        from comparison.protocol import add_comparison_metrics, write_json

        dataset = comparison["dataset"]
    predict = predictor(args, data, config, dataset)

    if args.history:
        result = {"history": args.history, "recommendations": predict(args.history)}
    else:
        if args.cohort:
            users = selected
        else:
            indices = list(range(len(data.users)))
            random.Random(args.seed).shuffle(indices)
            if args.test_users:
                indices = indices[:args.test_users]
            users = [data.users[index] for index in indices]
        entries = []
        for user in users:
            recommendations = predict(user["items"][:-1])
            ids = [row["item_id"] for row in recommendations]
            target = user["items"][-1]
            rank = ids.index(target) + 1 if target in ids else None
            entry = {"user_id": user["user_id"], "target_item": target, "rank": rank,
                     "recommendations": recommendations}
            entries.append(entry)
            print(json.dumps(entry), flush=True)
        result = {"users": len(entries), f"Recall@{args.top_k}": sum(row["rank"] is not None for row in entries) / len(entries),
                  f"NDCG@{args.top_k}": sum(1 / math.log2(row["rank"] + 1) if row["rank"] else 0 for row in entries) / len(entries),
                  "protocol": "full catalog, seen items excluded, autoregressive beam, no teacher forcing",
                  "policy_normalization": config.get("policy_normalization", "full"),
                  "beam_size": args.beam_size, "seed": args.seed, "entries": entries}
    result.update(scoring_settings(args.scoring_backend))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cohort:
        add_comparison_metrics(result, selected, data.item_ids, comparison, "OpenJev")
        if selection is not None:
            result.update(train_selection=selection["train_selection"],
                          validation_protocol=selection["validation_protocol"],
                          best_validation=selection["best_validation"])
        write_json(args.output, result)
    else:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "entries"}), flush=True)


if __name__ == "__main__":
    main()
