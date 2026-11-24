"""Read OpenJev's prepared MovieLens sequences and attach native TIGER SIDs."""

import json
import sys

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from experiment_tracking import sha256_file
from ml1m_eval import comparison_protocol


CODEBOOK_SIZE = 256
SID_LAYERS = 4
SOURCE_LAYERS = 3
HOLDOUT_OFFSETS = {"calibration": 3, "validation": 2, "test": 1}


def load_sequences(openjev_root, data_dir, comparison_data=None):
    sys.path.insert(0, str(openjev_root))
    from pipeline.data_preparation.build_jev_data import SequenceData

    data = SequenceData(data_dir, comparison_data)
    catalog_path = data_dir / "item_semantic_ids.jsonl"
    if sha256_file(catalog_path) != data.manifest["semantic_ids_sha256"]:
        raise ValueError("OpenJev catalog fingerprint differs from the sequence manifest")
    if (data.manifest["layers"] != SOURCE_LAYERS or data.manifest["choices"] != 255
            or data.manifest["max_history"] < 1
            or data.manifest["split_protocol"] !=
            "last=test, penultimate=validation, third-last=calibration, earlier=train"):
        raise ValueError("OpenJev data must use the three-layer ML-1M chronological split")
    with catalog_path.open(encoding="utf-8") as handle:
        catalog = list(map(json.loads, handle))
    catalog_ids = {row["item_id"] for row in catalog}
    if len(catalog_ids) != len(catalog) or not catalog_ids:
        raise ValueError("OpenJev catalog must contain unique movie IDs")
    if any(len(user["items"]) < 5 or not catalog_ids.issuperset(user["items"])
           for user in data.users):
        raise ValueError("Invalid OpenJev user sequence length or unknown movie ID")
    if (not data.users or len(data.users) != data.manifest["users"]
            or sum(len(user["items"]) for user in data.users) != data.manifest["interactions"]
            or any(data.count(split) != data.manifest["summary"][split] for split in HOLDOUT_OFFSETS)):
        raise ValueError("OpenJev sequence counts differ from the manifest")
    training = TrainingSequences(data.users, data.manifest, sequence_path=data.sequence_path)
    # Older manifests counted all prefix windows. New ones count only OpenJev's target per user.
    if "train_selection" in data.manifest:
        train_count = training.count("train")
    else:
        train_count = (data.manifest["interactions"] - 4 * len(data.users)) * SOURCE_LAYERS
    if train_count != data.manifest["summary"]["train"]:
        raise ValueError("OpenJev training count differs from the manifest")
    return training, catalog


class TrainingSequences:
    """Keep one shared next-item target per user."""

    def __init__(self, users, manifest, *, sequence_path=None):
        self.users = users
        self.manifest = manifest
        self.sequence_path = sequence_path
        self.train_pairs = len(users)

    def count(self, split):
        return len(self.users) * SOURCE_LAYERS


def load_comparison_sequences(directory, dataset):
    protocol = comparison_protocol()
    from comparison.features import sentence_catalog
    from comparison.prepare import load_prepared

    users, items, manifest = load_prepared(directory, dataset)
    data = TrainingSequences(users, {"max_history": protocol.MAX_HISTORY, "dataset": dataset})
    if data.train_pairs != manifest["train_pairs"]:
        raise ValueError("TIGER training positions differ from the shared dataset")
    return data, sentence_catalog(directory, dataset, items, manifest), manifest


def load_native_codes(code_path, embedding_path, catalog):
    item_ids = pd.read_parquet(embedding_path, columns=["ItemID"])["ItemID"]
    item_ids = [str(item) for item in item_ids]
    if item_ids != [row["item_id"] for row in catalog]:
        raise ValueError("TIGER embedding row order must match the OpenJev catalog")
    digits = np.load(code_path, allow_pickle=False)
    if (digits.shape != (len(item_ids), SID_LAYERS)
            or not np.issubdtype(digits.dtype, np.integer)
            or np.any(digits < 0) or np.any(digits >= CODEBOOK_SIZE)):
        raise ValueError("Expected four native TIGER SID digits per movie, each in [0, 255]")
    codes = dict(zip(item_ids, map(tuple, digits.tolist())))
    if len(set(codes.values())) != len(codes):
        raise ValueError("Native TIGER SID export must uniquely identify every movie")
    return codes


def token_offset(layer):
    return 1 + layer * CODEBOOK_SIZE


def model_tokens(code):
    return tuple(token_offset(layer) + digit for layer, digit in enumerate(code))


def history_tokens(history, item_tokens, max_history):
    encoded = [item_tokens[item] for item in history[-max_history:]]
    return [(0,) * SID_LAYERS] * (max_history - len(encoded)) + encoded


def holdout(user, split):
    position = -HOLDOUT_OFFSETS[split]
    return user["items"][:position], user["items"][position]


class TrainingPairs(Dataset):
    """Index one training pair per training position, with four TIGER labels."""

    def __init__(self, data, item_tokens):
        self.users = data.users
        self.train_position = comparison_protocol().train_position
        self.max_history = data.manifest["max_history"]
        self.item_tokens = item_tokens

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        position = self.train_position(user)
        items = user["items"]
        return {
            "history": history_tokens(items[max(0, position - self.max_history):position],
                                      self.item_tokens, self.max_history),
            "target": self.item_tokens[items[position]],
        }
