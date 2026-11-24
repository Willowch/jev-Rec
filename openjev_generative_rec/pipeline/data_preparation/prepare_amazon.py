"""把 Amazon 5-core 评论和商品元数据转换为 OpenJev 的紧凑序列。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from ..common import ROOT, enable_comparison, sha256
from .build_jev_data import TOKENS, read_ids, sequence_split_metadata, sid_layout
from .embedding_cache import prepare_embeddings

enable_comparison()
from comparison.prepare import load_metadata, load_prepared
from comparison.sources import amazon_metadata, amazon_sequences as read_sequences

DATASETS = ("Industrial_and_Scientific", "Office_Products")
MIN_ITEMS = 5
MAX_HISTORY = 20
MAX_TEXT_TOKENS = 128
SID_LAYERS = 3
SEED = 42


@contextmanager
def staged_directory(parent, prefix):
    path = parent.resolve() / f"{prefix}{uuid4().hex}"
    # Use the parent's ACL on Windows, including when running in a restricted token.
    path.mkdir(mode=0o755 if os.name == "nt" else 0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def openjev_metadata(rows):
    return {row["item_id"]: (row["title"], row["categories"] or row["main_category"])
            for row in rows if row["metadata_found"]}


def limit_text(text, tokenizer):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    if len(encoded["input_ids"]) <= MAX_TEXT_TOKENS:
        return text
    end = encoded["offset_mapping"][MAX_TEXT_TOKENS - 1][1]
    text = text[:end].rstrip()
    # A byte-level token can end inside a Unicode character. Verify the character-safe prefix.
    while len(tokenizer.encode(text, add_special_tokens=False, verbose=False)) > MAX_TEXT_TOKENS:
        text = text[:-1].rstrip()
    return text


def build_catalog(category, item_ids, metadata, tokenizer):
    catalog = []
    truncated = 0
    for item in sorted(item_ids):
        title, categories = metadata.get(item, ("", ""))
        title = title or f"ASIN {item}"
        categories = categories or category.replace("_", " ")
        text = f"{title}. Categories: {categories}."
        limited_text = limit_text(text, tokenizer)
        truncated += limited_text != text
        catalog.append({"item_id": item, "title": title, "categories": categories, "text": limited_text})
    stats = {"items_without_metadata": len(item_ids) - len(metadata),
             "items_without_title": len(item_ids) - sum(bool(title) for title, _ in metadata.values()),
             "item_texts_truncated": truncated}
    return catalog, stats


def write_jsonl(path, rows, *, ensure_ascii=True):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=ensure_ascii, separators=(",", ":")) + "\n")


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def encode_catalog(catalog, tokenizer, base_model, out, batch_size):
    import numpy as np
    import torch
    from tqdm import tqdm

    from .item_embeddings import load_encoder, mean_pool

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, total_parameters, model_type = load_encoder(str(base_model), device)
    embeddings = np.lib.format.open_memmap(
        out / "embeddings.npy", mode="w+", dtype=np.float32,
        shape=(len(catalog), model.config.hidden_size),
    )
    with torch.inference_mode():
        for start in tqdm(range(0, len(catalog), batch_size), desc="embedding"):
            texts = [row["text"] for row in catalog[start:start + batch_size]]
            batch = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TEXT_TOKENS,
                              return_tensors="pt").to(device)
            hidden = model(**batch, use_cache=False).last_hidden_state
            vectors = mean_pool(hidden, batch["attention_mask"]).cpu().numpy()
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 1e-12).any():
                raise ValueError("embedding contains non-finite or zero-norm vectors")
            embeddings[start:start + len(vectors)] = vectors / norms
    embeddings.flush()
    manifest = {"items": len(catalog), "dim": model.config.hidden_size, "device": device, "model": str(base_model),
                "model_type": model_type, "parameters": sum(p.numel() for p in model.parameters()),
                "full_model_parameters": total_parameters, "batch_size": batch_size, "max_length": MAX_TEXT_TOKENS,
                "dtype": str(next(model.parameters()).dtype), "text": "title + categories",
                "pooling": "attention-masked mean + L2 normalization", "trainable_parameters": 0,
                "chat_template": False}
    provenance = base_model / "download_manifest.json"
    if provenance.is_file():
        manifest["model_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    write_manifest(out / "manifest.json", manifest)


def verify_quantization(catalog, embeddings, semantic_ids):
    import numpy as np

    vectors = np.load(embeddings / "embeddings.npy", mmap_mode="r")
    codes = np.load(semantic_ids / "semantic_ids.npy", mmap_mode="r")
    if vectors.ndim != 2 or len(vectors) != len(catalog) or codes.shape != (len(catalog), SID_LAYERS):
        raise ValueError("embedding and SID row counts must match the item catalog")
    with (semantic_ids / "item_semantic_ids.jsonl").open(encoding="utf-8") as handle:
        for item, code, row in zip(catalog, codes, map(json.loads, handle), strict=True):
            tokens = row.pop("semantic_ids")
            if (row != item or any(not 0 <= label < len(TOKENS) for label in code)
                    or tokens[:SID_LAYERS] != [TOKENS[label] for label in code]):
                raise ValueError(f"catalog/embedding/SID alignment failed for {item['item_id']}")
    return sid_layout(read_ids(semantic_ids / "item_semantic_ids.jsonl"))[0]


def quantize(embeddings, semantic_ids, collision_policy, *, clusters=len(TOKENS)):
    subprocess.run([sys.executable, "-B", "-u", "-m", "pipeline.data_preparation.rq_kmeans",
                    "--embeddings", str(embeddings / "embeddings.npy"), "--item-meta", str(embeddings / "items.jsonl"),
                    "--out", str(semantic_ids), "--layers", str(SID_LAYERS), "--collision-policy", collision_policy,
                    "--clusters", str(clusters), "--seed", str(SEED), "--method", "sklearn"], cwd=ROOT, check=True)


def prepare_dataset(category: str, raw: Path, base_model: Path, *, batch_size: int,
                    comparison_data=None, output=None, collision_policy="allow", embedding_cache=None):
    reviews, meta = raw / f"{category}_5.json.gz", raw / f"meta_{category}.json.gz"
    output = output or raw / "openjev"
    if output.exists():
        raise FileExistsError(f"prepared data already exists: {output}")
    inputs = (base_model / "config.json",) if comparison_data else (reviews, meta, base_model / "config.json")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"required input is missing: {path}")

    if comparison_data is not None:
        sequences, items, prepared = load_prepared(comparison_data, category)
        item_ids = set(items)
        source_hashes = prepared["source"]
        metadata = openjev_metadata(load_metadata(comparison_data, items, prepared))
    else:
        sequences, review_stats = read_sequences(reviews)
        item_ids = {item for row in sequences for item in row["items"]}
        source_hashes = {"ratings_sha256": sha256(reviews), "metadata_sha256": sha256(meta), **review_stats}
        metadata = openjev_metadata(amazon_metadata(meta, item_ids))
    if len(item_ids) < len(TOKENS):
        raise ValueError(f"{category}: RQ-KMeans requires at least {len(TOKENS)} eligible items")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    catalog, catalog_stats = build_catalog(category, item_ids, metadata, tokenizer)
    user_count = len(sequences)
    interactions = sum(len(row["items"]) for row in sequences)

    # Publish only once all stages complete. The staging directory is on the same filesystem.
    output.parent.mkdir(parents=True, exist_ok=True)
    with staged_directory(output.parent, ".openjev-") as temporary:
        stage = temporary / "openjev"
        embeddings, semantic_ids, sequence_dir = (stage / name for name in ("embeddings", "semantic_ids", "sequences"))
        embeddings.mkdir(parents=True)
        sequence_dir.mkdir()
        if comparison_data is None:
            write_jsonl(sequence_dir / "sequences.jsonl", sequences)

        def encode():
            write_jsonl(embeddings / "items.jsonl", catalog, ensure_ascii=False)
            encode_catalog(catalog, tokenizer, base_model, embeddings, batch_size)

        if comparison_data is not None:
            prepare_embeddings(embeddings, catalog, prepared["identity"], base_model, batch_size, MAX_TEXT_TOKENS,
                               encode, embedding_cache)
        else:
            encode()
        quantize(embeddings, semantic_ids, collision_policy)
        depth = verify_quantization(catalog, embeddings, semantic_ids)
        sid_path = sequence_dir / "item_semantic_ids.jsonl"
        (semantic_ids / sid_path.name).rename(sid_path)
        manifest = {
            "dataset": category, "users": user_count, "items": len(item_ids), "interactions": interactions,
            "max_history": MAX_HISTORY, "min_items": MIN_ITEMS, "min_rating": 0.0,
            "layers": depth, "choices": len(TOKENS),
            **source_hashes,
            "semantic_ids_sha256": sha256(sid_path),
            "source_files": {"reviews": reviews.name, "metadata": meta.name},
            "split_protocol": "last=test, penultimate=validation, third-last=calibration, earlier=train",
            "ordering": "earliest review per user and ASIN; sort by timestamp then ASIN",
            "group_unit": f"user-and-target-position; all {depth} token decisions remain together",
            "record_source": {
                "group_prefix": f"amazon:{category}", "name": f"amazon-{category}-semantic-id",
                "source_url": f"file:data/{category}/{reviews.name}", "license": "Unspecified for user-provided files",
                "split_policy": "per_user_first_asin_chronological_train_cal_val_test",
            },
            **sequence_split_metadata(user_count, depth),
            **catalog_stats,
        }
        if comparison_data is not None:
            manifest["comparison_identity"] = prepared["identity"]
        write_manifest(sequence_dir / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(f"prepared data already exists: {output}")
        stage.rename(output)

    print(json.dumps(manifest, indent=2), flush=True)
