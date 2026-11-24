"""Verify the inputs and files of a reusable comparison embedding directory."""

import hashlib
import json
import shutil

from ..common import sha256


def encoder_fingerprint(model):
    digest = hashlib.sha256()
    for path in sorted(model.iterdir()):
        if path.is_file() and path.suffix in {".json", ".model", ".safetensors", ".bin", ".txt", ".tiktoken"}:
            digest.update(json.dumps([path.name, sha256(path)]).encode("utf-8"))
    return digest.hexdigest()


def prepare_embeddings(output, catalog, identity, model, batch_size, max_length, encode, cache=None):
    """Use preprocessed metadata in its exact embedding order and validate both cached files."""
    inputs = {"dataset": identity["dataset"], "catalog_sha256": identity["catalog_sha256"],
              "items_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode("utf-8")).hexdigest(),
              "encoder_sha256": encoder_fingerprint(model), "max_length": max_length,
              "batch_size": batch_size, "encoding": "attention-masked-mean-l2-v1"}
    names = ("items.jsonl", "embeddings.npy")
    if cache is None:
        encode()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["cache_inputs"] = inputs
        manifest["cache_files"] = {name: sha256(output / name) for name in names}
    else:
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("cache_inputs") != inputs:
            raise ValueError(f"Embedding cache does not match the comparison inputs: {cache}")
        hashes = manifest.get("cache_files", {})
        for name in names:
            shutil.copyfile(cache / name, output / name)
            if sha256(output / name) != hashes.get(name):
                raise ValueError(f"Embedding cache file fingerprint differs: {cache / name}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
