"""Build OpenJev embeddings, identity SIDs and sequence metadata from shared comparison data."""

import argparse
import subprocess
import sys
from pathlib import Path

from ..common import ROOT
from .embedding_cache import prepare_embeddings
from .prepare_amazon import (MAX_TEXT_TOKENS, encode_catalog, prepare_dataset, quantize, verify_quantization,
                             write_jsonl)
from comparison.prepare import load_metadata, load_prepared
from comparison.protocol import COMPARISON_DATASETS, PILOT_DATASET, staged_output
from comparison.sources import raw_directory


def prepare_movielens(dataset, shared, model, output, batch_size, cache, *, clusters=255):
    from transformers import AutoTokenizer

    _, items, manifest = load_prepared(shared, dataset)
    catalog = load_metadata(shared, items, manifest)
    with staged_output(output) as stage:
        embeddings = stage / "embeddings"
        semantic_ids = stage / "semantic_ids"
        embeddings.mkdir()

        def encode():
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            tokenizer.padding_side = "right"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            write_jsonl(embeddings / "items.jsonl", catalog, ensure_ascii=False)
            encode_catalog(catalog, tokenizer, model, embeddings, batch_size)

        prepare_embeddings(embeddings, catalog, manifest["identity"], model, batch_size, MAX_TEXT_TOKENS, encode, cache)
        quantize(embeddings, semantic_ids, "append", clusters=clusters)
        verify_quantization(catalog, embeddings, semantic_ids)
        subprocess.run([sys.executable, "-B", "-u", "-m", "pipeline.data_preparation.build_jev_data",
                        "--dataset", dataset, "--comparison-data", str(shared),
                        "--semantic-ids", str(semantic_ids / "item_semantic_ids.jsonl"),
                        "--out", str(stage / "sequences")], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=COMPARISON_DATASETS, required=True)
    for name in ("comparison-data", "base-model", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=255, help="Smaller SID codebook for the ML-100K pilot")
    args = parser.parse_args()
    model = args.base_model.resolve()
    if args.batch_size < 1 or not (model / "config.json").is_file():
        parser.error("provide a positive batch size and a local base model directory")
    if not 2 <= args.clusters <= 255 or (args.clusters != 255 and args.dataset != PILOT_DATASET):
        parser.error("Only the MovieLens-100K pilot supports smaller codebooks (2 to 255 clusters)")
    if args.dataset in ("ml1m", PILOT_DATASET):
        prepare_movielens(args.dataset, args.comparison_data.resolve(), model, args.output,
                          args.batch_size, args.embedding_cache, clusters=args.clusters)
    else:
        prepare_dataset(args.dataset, raw_directory(args.dataset), model, batch_size=args.batch_size,
                        comparison_data=args.comparison_data.resolve(), output=args.output,
                        collision_policy="append", embedding_cache=args.embedding_cache)


if __name__ == "__main__":
    main()
