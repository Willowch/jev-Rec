"""Prepare Amazon 5-core data under the TIGER ML-1M split protocol."""

import gzip
import json
import os
import random

from experiment_tracking import sha256_file, write_json
from ml1m_data import HOLDOUT_OFFSETS, TrainingSequences
from ml1m_eval import comparison_protocol


DATASETS = ("Industrial_and_Scientific", "Office_Products")
SPLIT_PROTOCOL = "last=test, penultimate=validation, third-last=calibration, earlier=train"


def json_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def product_catalog(metadata, items):
    comparison_protocol()
    from comparison.sources import amazon_metadata

    rows = amazon_metadata(metadata, set(items))
    return ([{"item_id": row["item_id"], "text": row["tiger_text"]} for row in rows],
            sum(row["tiger_fallback"] for row in rows))


def prepare_data(dataset, raw_dir, output, *, max_history, min_rating, smoke, smoke_users, seed,
                 comparison_data=None):
    reviews = raw_dir / f"{dataset}_5.json.gz"
    metadata = raw_dir / f"meta_{dataset}.json.gz"
    raw_files = (reviews, metadata) if comparison_data is None else ()
    for path in raw_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    identity = None
    if comparison_data is not None:
        protocol = comparison_protocol()
        from comparison.prepare import load_metadata, load_prepared

        if max_history != protocol.MAX_HISTORY or min_rating != 0.0:
            raise ValueError("comparison data requires the OpenJev history limit, all ratings and all users")
        sequences, items, source = load_prepared(comparison_data, dataset)
        asins = set(items)
        eligible_users = len(sequences)
        identity = source["identity"]
        metadata_rows = load_metadata(comparison_data, items, source)
        missing = sum(row["tiger_fallback"] for row in metadata_rows)
    else:
        comparison_protocol()
        from comparison.sources import amazon_sequences

        print(f"Reading {reviews.name}", flush=True)
        sequences, _ = amazon_sequences(reviews, min_rating=min_rating)
        if not sequences:
            raise ValueError("No users with at least five interactions")
        eligible_users = len(sequences)
        if smoke and eligible_users > smoke_users:
            selected = sorted(random.Random(seed).sample(range(eligible_users), smoke_users))
            sequences = [sequences[index] for index in selected]
        asins = {item for user in sequences for item in user["items"]}

    seq_path = output / "sequences.jsonl"
    catalog_path = output / "items.jsonl"
    if identity is None:
        print(f"Reading {metadata.name} for {len(asins)} catalog items", flush=True)
        catalog, missing = product_catalog(metadata, asins)
        write_rows(seq_path, sequences)
        write_rows(catalog_path, catalog)
    else:
        seq_path = comparison_data / "sequences.jsonl"
        catalog_path = comparison_data / "metadata.jsonl"
    data = TrainingSequences(sequences, {"max_history": max_history})
    manifest = {
        "dataset": dataset,
        "split_protocol": SPLIT_PROTOCOL,
        "timestamp_tie_break": "lexicographic ASIN",
        "max_history": max_history,
        "min_rating": min_rating,
        "smoke": smoke,
        "seed": seed,
        "eligible_users": eligible_users,
        "users": len(sequences),
        "interactions": sum(len(user["items"]) for user in sequences),
        "catalog_size": len(asins),
        "fallback_text_items": missing,
        "train_pairs": data.train_pairs,
        "train_selection": comparison_protocol().TRAIN_SELECTION,
        "summary": {split: data.count(split) for split in ("train", *HOLDOUT_OFFSETS)},
        "raw_sha256": {path.name: sha256_file(path) for path in raw_files},
        "sequences_sha256": sha256_file(seq_path),
        "catalog_sha256": sha256_file(catalog_path),
    }
    if identity is not None:
        manifest["comparison_identity"] = identity
        manifest["comparison_data"] = os.path.relpath(comparison_data.resolve(), output.resolve())
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(sequences)}/{eligible_users} users, {len(asins)} items, "
          f"{manifest['train_pairs']} train pairs", flush=True)


def load_data(directory, dataset):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset or manifest["split_protocol"] != SPLIT_PROTOCOL:
        raise ValueError(f"Data in {directory} does not match {dataset} and its split protocol")
    if reference := manifest.get("comparison_data"):
        comparison_protocol()
        from comparison.features import sentence_catalog
        from comparison.prepare import load_prepared

        shared = (directory / reference).resolve()
        sequences, items, prepared = load_prepared(shared, dataset)
        if (prepared["identity"] != manifest["comparison_identity"]
                or prepared["metadata_sha256"] != manifest["catalog_sha256"]
                or sha256_file(shared / "sequences.jsonl") != manifest["sequences_sha256"]):
            raise ValueError("Shared TIGER data differs from the prepared run")
        data = TrainingSequences(sequences, manifest)
        if (data.train_pairs != prepared["train_pairs"]
                or ("train_selection" in manifest and data.train_pairs != manifest["train_pairs"])):
            raise ValueError("TIGER training positions differ from the shared dataset")
        return data, sentence_catalog(shared, dataset, items, prepared)
    for name, key in (("sequences.jsonl", "sequences_sha256"), ("items.jsonl", "catalog_sha256")):
        if sha256_file(directory / name) != manifest[key]:
            raise ValueError(f"{name} differs from its data manifest")
    data = TrainingSequences(list(json_rows(directory / "sequences.jsonl")), manifest)
    catalog = list(json_rows(directory / "items.jsonl"))
    return data, catalog
