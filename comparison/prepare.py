"""Prepare canonical next-item data and freeze the evaluation cohorts."""

import argparse
import json
from pathlib import Path

from .protocol import (COMPARISON_DATASETS, COUNTS, MAX_HISTORY, PILOT_DATASET, PILOT_ITEMS, PILOT_USERS,
                       ROOT, SEED, SPLIT_PROTOCOL,
                       dataset_identity, item_key, read_json, read_rows, sampled_users,
                       select_cohort_users, sha256, staged_output, TRAIN_SELECTION, user_targets, write_json)
from .sources import amazon_metadata, amazon_sequences, ml100k_source, ml1m_movies, ml1m_sequences, raw_directory


def read_source(dataset, openjev_root, raw_root=None):
    data = openjev_root / "data"
    raw = raw_directory(dataset, raw_root)
    if dataset == PILOT_DATASET:
        archive = raw / "ml-100k.zip"
        all_users, metadata = ml100k_source(archive)
        users = sorted(sampled_users(all_users, PILOT_USERS), key=lambda row: int(row["user_id"]))
        return users, [row["item_id"] for row in metadata], {
            "ratings_sha256": sha256(archive), "pilot_seed": SEED, "raw_users": len(all_users),
        }, metadata, None
    if dataset == "ml1m":
        directory = data / "ml1m_qwen_sequences"
        archive = raw / "ml-1m.zip"
        if not archive.exists() and raw_root is None:
            archive = data / "ml-1m/ml-1m.zip"
        if raw_root is not None or not (directory / "sequences.jsonl").is_file():
            metadata = sorted(ml1m_movies(archive), key=lambda row: int(row["item_id"]))
            items = {row["item_id"] for row in metadata}
            return list(ml1m_sequences(archive, items)), items, {"ratings_sha256": sha256(archive)}, metadata, None
    else:
        if not raw.is_dir() and raw_root is None:
            raw = data / dataset
        reviews = raw / f"{dataset}_5.json.gz"
        metadata_path = raw / f"meta_{dataset}.json.gz"
        users, stats = amazon_sequences(reviews)
        items = {item for user in users for item in user["items"]}
        return users, items, {"ratings_sha256": sha256(reviews), "metadata_sha256": sha256(metadata_path),
                              **stats}, amazon_metadata(metadata_path, items), None

    manifest = read_json(directory / "manifest.json")
    if (manifest.get("dataset", "ml1m") != dataset or manifest["max_history"] != MAX_HISTORY
            or manifest["split_protocol"] != SPLIT_PROTOCOL):
        raise ValueError(f"{directory}: expected OpenJev {dataset} data with the comparison split")
    if archive.is_file() and sha256(archive) != manifest["ratings_sha256"]:
        raise ValueError(f"{archive}: raw ratings fingerprint differs from the existing ML-1M sequences")
    item_path = directory / "item_semantic_ids.jsonl"
    catalog_hash = sha256(item_path)
    if catalog_hash != manifest["semantic_ids_sha256"]:
        raise ValueError(f"{item_path}: catalog fingerprint mismatch")
    metadata = [{key: value for key, value in row.items() if key != "semantic_ids"}
                for row in read_rows(item_path)]
    items = [row["item_id"] for row in metadata]
    if len(items) != len(set(items)):
        raise ValueError(f"{item_path}: duplicate item IDs")
    sequences = directory / "sequences.jsonl"
    return read_rows(sequences), items, {
        "ratings_sha256": manifest["ratings_sha256"], "semantic_ids_sha256": catalog_hash}, metadata, sequences


def validate_sequences(dataset, users, items):
    catalog = set(items)
    user_ids = [user["user_id"] for user in users]
    if user_ids != sorted(set(user_ids), key=lambda uid: item_key(dataset, uid)):
        raise ValueError("users must be unique and in the OpenJev source order")
    for user in users:
        events = user["items"]
        timestamps = user["timestamps"]
        if (len(events) < 5 or len(events) != len(timestamps) or not catalog.issuperset(events)
                or (dataset != "ml1m" and len(events) != len(set(events)))):
            raise ValueError(f"invalid OpenJev sequence for {user['user_id']}")
        timeline = list(zip(timestamps, (item_key(dataset, item) for item in events)))
        if timeline != sorted(timeline):
            raise ValueError(f"sequence out of chronological order for {user['user_id']}")
    actual = len(users), len(catalog), len(users)
    expected = (PILOT_USERS, PILOT_ITEMS, PILOT_USERS) if dataset == PILOT_DATASET else COUNTS[dataset]
    if actual != expected:
        raise ValueError(f"{dataset}: users/catalog/train pairs {actual}, expected {expected}")


def prepare(dataset, openjev_root, output, raw_root=None):
    users, items, source, metadata, existing_sequences = read_source(dataset, openjev_root, raw_root)
    validate_sequences(dataset, users, items)
    items = sorted(items, key=lambda item: item_key(dataset, item))
    manifest = {"identity": dataset_identity(dataset, users, items, MAX_HISTORY), "source": source,
                "users": len(users), "items": len(items), "train_pairs": len(users),
                "train_selection": TRAIN_SELECTION}
    with staged_output(output) as stage:
        sequences = stage / "sequences.jsonl"
        if existing_sequences is not None:
            sequences.write_bytes(existing_sequences.read_bytes())
        else:
            write_rows(sequences, users)
        write_json(stage / "items.json", items)
        item_metadata = stage / "metadata.jsonl"
        write_rows(item_metadata, metadata)
        manifest["metadata_sha256"] = sha256(item_metadata)
        write_json(stage / "manifest.json", manifest)
    return manifest


def load_prepared(directory, dataset):
    manifest = read_json(directory / "manifest.json")
    users = read_rows(directory / "sequences.jsonl")
    items = read_json(directory / "items.json")
    if manifest["identity"] != dataset_identity(dataset, users, items, MAX_HISTORY):
        raise ValueError(f"{directory}: sequences or catalog differ from the preparation manifest")
    # The sequence identity and cohort stay valid for previously prepared directories.
    if "train_selection" not in manifest:
        manifest = {**manifest, "train_selection": TRAIN_SELECTION, "train_pairs": len(users)}
    if manifest["train_selection"] != TRAIN_SELECTION or manifest["train_pairs"] != len(users):
        raise ValueError(f"{directory}: training targets differ from the shared protocol")
    return users, items, manifest


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def load_metadata(directory, items, manifest):
    source = directory / "metadata.jsonl"
    if sha256(source) != manifest["metadata_sha256"]:
        raise ValueError(f"{source}: product metadata fingerprint differs")
    rows = read_rows(source)
    by_id = {row["item_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(items):
        raise ValueError("Shared metadata and item catalog differ")
    return [by_id[item] for item in items]


def historical_entries(path):
    if path.suffix == ".json":
        result = read_json(path)
        if result["users"] != COUNTS["ml1m"][0]:
            raise ValueError("expected the full 6,040-user OpenJev ML-1M result")
        return result["entries"]
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.startswith('{"user_id":')]


def make_cohort(dataset, data, output, historical=None, smoke=False):
    users, _, manifest = load_prepared(data, dataset)
    if historical is not None and (dataset != "ml1m" or smoke):
        raise ValueError("the historical result is only used for the formal ML-1M cohort")
    if dataset == PILOT_DATASET:
        if smoke or len(users) != PILOT_USERS:
            raise ValueError("the MovieLens-100K pilot uses the full prepared set of 256 users")
        selected = users
        provenance = {"seed": SEED}
        scope = "pilot"
    elif smoke:
        selected = sampled_users(users, 8)
        provenance = {"seed": SEED}
        scope = "smoke"
    elif historical is not None:
        selected = select_cohort_users(users, historical_entries(historical), COUNTS[dataset][0])
        provenance = {"openjev_evaluation_sha256": sha256(historical)}
        scope = "formal"
    else:
        if len(users) != COUNTS[dataset][0]:
            raise ValueError(f"the formal {dataset} cohort requires all {COUNTS[dataset][0]} shared users")
        selected = users
        provenance = {}
        scope = "formal"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"identity": manifest["identity"], "scope": scope,
                        **provenance, "entries": user_targets(selected)})
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    data = stages.add_parser("data")
    data.add_argument("--openjev-root", type=Path, default=ROOT / "openjev_generative_rec")
    data.add_argument("--raw-root", type=Path, help="Directory holding one raw subdirectory per dataset")
    cohort = stages.add_parser("cohort")
    cohort.add_argument("--data-dir", type=Path, required=True)
    cohort.add_argument("--openjev-evaluation", type=Path)
    cohort.add_argument("--smoke", action="store_true")
    for stage in (data, cohort):
        stage.add_argument("--dataset", choices=COMPARISON_DATASETS, required=True)
        stage.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "data":
        print(json.dumps(prepare(args.dataset, args.openjev_root, args.output, args.raw_root), indent=2))
    else:
        count = make_cohort(args.dataset, args.data_dir, args.output, args.openjev_evaluation, args.smoke)
        print(f"{args.dataset}: saved {count} test users to {args.output}")


if __name__ == "__main__":
    main()
