"""ML-1M 时间切分；紧凑保存序列，按需展开为 OpenJev 的 255-way Choice。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..common import enable_comparison, sha256

TOKENS = [f"sid_{i:03d}" for i in range(255)]
SEMANTIC_LAYERS = 3
TOKEN_SET = frozenset(TOKENS)
IDENTITY_TOKEN = re.compile(r"<d_(0|[1-9][0-9]*)>")
TARGET_OFFSETS = {"train": 4, "calibration": 3, "validation": 2, "test": 1}
SPLITS = (*TARGET_OFFSETS, "ood")
TRAIN_SELECTION = "last_before_calibration"


def sequence_split_metadata(user_count, depth):
    return {
        "train_selection": TRAIN_SELECTION,
        "train_positions": user_count,
        "summary": {**dict.fromkeys(TARGET_OFFSETS, user_count * depth), "ood": 0},
    }


def read_ratings(path):
    enable_comparison()
    from comparison.sources import ml1m_ratings

    return ml1m_ratings(path)


def read_ids(path):
    item_ids = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            item = row["item_id"]
            if item in item_ids:
                raise ValueError(f"SID catalog contains a duplicate item ID: {item}")
            item_ids[item] = row["semantic_ids"]
    return item_ids


def sid_layout(item_ids):
    depths = {len(code) for code in item_ids.values()}
    if depths not in ({SEMANTIC_LAYERS}, {SEMANTIC_LAYERS + 1}):
        raise ValueError("SID catalog must uniformly contain three semantic tokens and an optional identity token")
    depth = next(iter(depths))
    identities = set()
    codes = set()
    for code in item_ids.values():
        if any(token not in TOKEN_SET for token in code[:SEMANTIC_LAYERS]):
            raise ValueError(f"Invalid semantic SID: {code}")
        if depth > SEMANTIC_LAYERS:
            if not IDENTITY_TOKEN.fullmatch(code[-1]) or tuple(code) in codes:
                raise ValueError(f"Invalid or duplicate identity SID: {code}")
            identities.add(code[-1])
            codes.add(tuple(code))
    return depth, sorted(identities, key=lambda token: int(token[3:-1]))


def input_record(history_ids, prefix, history_items=None, *, options=TOKENS):
    row = {"state": {"history": " ".join("/".join(ids) for ids in history_ids),
                      "target_prefix": "/".join(prefix) or "<empty>",
                      "semantic_id_layer": len(prefix) + 1},
            "question": f"Predict semantic ID at layer {len(prefix) + 1} of the next item.",
           "kind": "choice", "options": options}
    if history_items is not None:
        row["metadata"] = {"history_items": list(history_items)}
    return row


class CatalogPolicy:
    """Candidate support for each prefix, including the complete item catalog."""
    def __init__(self, item_ids):
        self.depth, self.identity_tokens = sid_layout(item_ids)
        self.items_by_code, choices = {}, {}
        for item, ids in item_ids.items():
            code = tuple(ids)
            self.items_by_code.setdefault(code, []).append(item)
            for level, token in enumerate(code):
                choices.setdefault(code[:level], set()).add(token)
        self.options = {prefix: sorted(tokens) for prefix, tokens in choices.items()}

    def row(self, history_ids, prefix, history_items=None):
        return input_record(history_ids, prefix, history_items, options=self.options[tuple(prefix)])


class SequenceData:
    """每位用户在各时间切分中提供一个目标，按需生成 SID 层记录。"""
    def __init__(self, directory, comparison_data=None):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.record_source = self.manifest.get("record_source", {
            "group_prefix": "ml1m",
            "name": "ml1m-semantic-id",
            "source_url": "https://grouplens.org/datasets/movielens/1m/",
            "license": "MovieLens research terms",
            "split_policy": "per_user_chronological_train_cal_val_test",
        })
        self.item_ids = read_ids(directory / "item_semantic_ids.jsonl")
        self.catalog = CatalogPolicy(self.item_ids)
        self.depth = self.catalog.depth
        if self.manifest.get("layers", SEMANTIC_LAYERS) != self.depth:
            raise ValueError("Sequence manifest SID depth differs from the catalog")
        original = directory / "sequences.jsonl"
        self.sequence_path = original
        self.comparison_identity = None
        if comparison_data is not None or not original.is_file():
            enable_comparison()
            from comparison.prepare import load_prepared
            from comparison.protocol import DATA_ROOT, MAX_HISTORY, SPLIT_PROTOCOL, dataset_identity, read_rows

            dataset = self.manifest.get("dataset", "ml1m")
            shared = Path(comparison_data) if comparison_data is not None else DATA_ROOT / dataset
            users, items, prepared = load_prepared(shared, dataset)
            identity = prepared["identity"]
            if (set(items) != set(self.item_ids) or self.manifest["max_history"] != MAX_HISTORY
                    or self.manifest["split_protocol"] != SPLIT_PROTOCOL
                    or prepared["source"]["ratings_sha256"] != self.manifest["ratings_sha256"]
                    or self.manifest.get("comparison_identity", identity) != identity):
                raise ValueError("Shared sequences do not match the OpenJev SID catalog and source manifest")
            if original.is_file() and dataset_identity(dataset, read_rows(original), items, MAX_HISTORY) != identity:
                raise ValueError("Shared sequences differ from the original OpenJev sequences")
            self.sequence_path = shared / "sequences.jsonl"
            self.comparison_identity = identity
            self.users = users
        else:
            with original.open(encoding="utf-8") as handle:
                self.users = list(map(json.loads, handle))
            if any(len(user["items"]) <= TARGET_OFFSETS["train"] for user in self.users):
                raise ValueError("OpenJev sequences require at least five interactions per user")
        self.train_positions = len(self.users)
        self.train_decisions = self.train_positions * self.depth

    def row(self, split, index, *, catalog=False):
        user_index, level = divmod(index, self.depth)
        user = self.users[user_index]
        items = user["items"]
        position = len(items) - TARGET_OFFSETS[split]
        target_item = items[position]
        target_ids = self.item_ids[target_item]
        history = items[max(0, position - self.manifest["max_history"]):position]
        prefix = target_ids[:level]
        options = self.catalog.options[tuple(prefix)] if catalog or level >= SEMANTIC_LAYERS else TOKENS
        row = input_record([self.item_ids[item] for item in history], prefix, options=options)
        group = f"{self.record_source['group_prefix']}:{user['user_id']}:{position}"
        row.update(id=f"{group}:{level}", group_id=group, split=split, source=self.record_source["name"],
                   target=[float(token == target_ids[level]) for token in options],
                   metadata={"user_id": user["user_id"], "target_item": target_item,
                             "position": position, "semantic_layer": level + 1,
                             "history_items": history, "target_basis": "next_observed_item",
                             "provenance": {"type": "import", "input_sha256": self.manifest["ratings_sha256"],
                                            "source_url": self.record_source["source_url"],
                                            "license": self.record_source["license"], "original_id": group,
                                            "split_policy": self.record_source["split_policy"]}})
        return row

    def count(self, split):
        return self.train_decisions if split == "train" else 0 if split == "ood" else self.depth * len(self.users)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ratings", type=Path)
    source.add_argument("--comparison-data", type=Path)
    parser.add_argument("--dataset", choices=("ml1m", "ml100k"), default="ml1m",
                        help="Dataset for --comparison-data")
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--export-limit", type=int, default=0, help="每 split 导出少量官方格式 JSONL 供校验；0 不导出")
    args = parser.parse_args()
    if args.max_history < 1:
        raise ValueError("max-history must be positive")
    if args.dataset == "ml100k" and args.comparison_data is None:
        parser.error("MovieLens-100K uses the shared prepared comparison data")
    args.out.mkdir(parents=True, exist_ok=True)
    item_ids = read_ids(args.semantic_ids)
    depth, _ = sid_layout(item_ids)
    enable_comparison()
    from comparison.prepare import load_prepared, write_rows
    from comparison.protocol import MAX_HISTORY, SPLIT_PROTOCOL
    from comparison.sources import ml1m_sequences

    if args.comparison_data is not None:
        users, items, prepared = load_prepared(args.comparison_data, args.dataset)
        if set(items) != set(item_ids) or args.max_history != MAX_HISTORY or args.min_rating != 0.0:
            raise ValueError("Shared MovieLens data requires the full SID catalog, all ratings and history length 20")
        ratings_hash = prepared["source"]["ratings_sha256"]
    else:
        users = list(ml1m_sequences(args.ratings, item_ids, args.min_rating))
        ratings_hash = sha256(args.ratings)
        write_rows(args.out / "sequences.jsonl", users)
    count = len(users)
    interactions = sum(len(user["items"]) for user in users)
    sid_path = args.out / "item_semantic_ids.jsonl"
    if sid_path.resolve() != args.semantic_ids.resolve():
        sid_path.write_bytes(args.semantic_ids.read_bytes())
    manifest = {"users": count, "interactions": interactions, "max_history": args.max_history,
                "min_rating": args.min_rating, "layers": depth, "choices": len(TOKENS),
                "ratings_sha256": ratings_hash, "semantic_ids_sha256": sha256(sid_path),
                "split_protocol": SPLIT_PROTOCOL,
                "group_unit": f"user-and-target-position; all {depth} token decisions remain together",
                **sequence_split_metadata(count, depth)}
    if args.comparison_data is not None:
        manifest["comparison_identity"] = prepared["identity"]
    if args.dataset == "ml100k":
        manifest["dataset"] = args.dataset
        manifest["record_source"] = {"group_prefix": args.dataset, "name": "ml100k-semantic-id",
                                     "source_url": "https://grouplens.org/datasets/movielens/100k/",
                                     "license": "MovieLens research terms",
                                     "split_policy": "per_user_chronological_train_cal_val_test"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.export_limit:
        dataset = SequenceData(args.out, args.comparison_data)
        export = args.out / "schema_sample"
        export.mkdir(exist_ok=True)
        for split in SPLITS:
            with (export / f"{split}.jsonl").open("w") as handle:
                for i in range(min(dataset.count(split), args.export_limit)):
                    handle.write(json.dumps(dataset.row(split, i), separators=(",", ":")) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
