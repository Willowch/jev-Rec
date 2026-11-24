"""One checked Sentence-T5 feature table per dataset and encoder snapshot."""

import hashlib
import json
from pathlib import Path

from .prepare import load_metadata
from .protocol import PILOT_DATASET, read_json, sha256, staged_output, write_json


def sentence_catalog(data_dir, dataset, items, manifest):
    rows = load_metadata(data_dir, items, manifest)
    if dataset in ("ml1m", PILOT_DATASET):
        return rows
    return [{"item_id": row["item_id"], "text": row["tiger_text"]} for row in rows]


def sentence_t5(data_dir, dataset, items, manifest, snapshot, device, batch_size):
    catalog = sentence_catalog(data_dir, dataset, items, manifest)
    snapshot = Path(snapshot).resolve()
    if not (snapshot / "modules.json").is_file():
        raise FileNotFoundError(f"Missing local Sentence-T5 snapshot: {snapshot}")
    fingerprint = {
        "dataset": dataset, "catalog_sha256": manifest["identity"]["catalog_sha256"],
        "texts_sha256": hashlib.sha256(json.dumps([row["text"] for row in catalog],
                                                  ensure_ascii=False).encode()).hexdigest(),
        "snapshot_sha256": {path.relative_to(snapshot).as_posix(): sha256(path)
                            for path in sorted(snapshot.rglob("*")) if path.is_file()},
        "normalize_embeddings": False,
    }
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    destination = data_dir / "features" / "sentence-t5" / key
    if not destination.exists():
        try:
            with staged_output(destination) as stage:
                encode_sentence_t5(catalog, dataset, snapshot, device, batch_size, stage / "item_emb.parquet")
                write_json(stage / "manifest.json", {"encoding": fingerprint, "batch_size": batch_size,
                                                     "sha256": sha256(stage / "item_emb.parquet")})
        except FileExistsError:
            if not destination.is_dir():
                raise
    recorded = read_json(destination / "manifest.json")
    path = destination / "item_emb.parquet"
    if recorded["encoding"] != fingerprint or sha256(path) != recorded["sha256"]:
        raise ValueError("Shared Sentence-T5 features differ from their manifest")
    return path, recorded


def encode_sentence_t5(catalog, dataset, snapshot, device, batch_size, output):
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    print(f"Sentence-T5: encoding {len(catalog)} items", flush=True)
    encoder = SentenceTransformer(str(snapshot), device=str(device), local_files_only=True)
    vectors = encoder.encode([row["text"] for row in catalog], batch_size=batch_size,
                             normalize_embeddings=False, show_progress_bar=True, convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(catalog) or not np.isfinite(vectors).all():
        raise ValueError("Invalid Sentence-T5 feature matrix")
    ids = [int(row["item_id"]) if dataset in ("ml1m", PILOT_DATASET) else row["item_id"] for row in catalog]
    pd.DataFrame({"ItemID": ids, "embedding": list(vectors)}).to_parquet(output, index=False)


def read_sentence_vectors(path, items):
    import numpy as np
    import pandas as pd

    table = pd.read_parquet(path, columns=["ItemID", "embedding"])
    if list(map(str, table["ItemID"])) != items:
        raise ValueError("Sentence-T5 feature rows differ from the ordered catalog")
    matrix = np.stack(table["embedding"]).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Sentence-T5 features contain non-finite values")
    return matrix
