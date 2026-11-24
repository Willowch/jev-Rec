"""Freeze dataset identity and evaluate a shared set of next-item targets."""

import hashlib
import json
import math
import os
import random
import shutil
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "comparison" / "data"
DATASETS = ("ml1m", "Industrial_and_Scientific", "Office_Products")
PILOT_DATASET = "ml100k"
COMPARISON_DATASETS = (*DATASETS, PILOT_DATASET)
PILOT_USERS = 256
PILOT_ITEMS = 1682
PILOT_VALIDATION_USERS = 64
COUNTS = {"ml1m": (6040, 3883, 6040),
          "Industrial_and_Scientific": (9331, 5285, 9331),
          "Office_Products": (88383, 27917, 88383)}
MAX_HISTORY = 20
TOP_K = 20
VALIDATION_K = 10
VALIDATION_INTERVAL = 5
METRIC_CUTOFFS = (5, VALIDATION_K, TOP_K)
SEED = 42
HOLDOUT_OFFSETS = {"calibration": 3, "validation": 2, "test": 1}
SPLIT_PROTOCOL = "last=test, penultimate=validation, third-last=calibration, earlier=train"
TRAIN_SELECTION = "last_before_calibration"
_EVALUATION_HEADER = '{"entries":[\n'


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(map(json.loads, handle))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def staged_output(destination):
    """Publish a complete directory without replacing a previous run."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".comparison-{uuid4().hex}"
    stage.mkdir()
    try:
        yield stage
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def item_key(dataset, item):
    return int(item) if dataset in ("ml1m", PILOT_DATASET) else item


def dataset_identity(dataset, users, catalog, max_history):
    if dataset not in COMPARISON_DATASETS or max_history != MAX_HISTORY:
        raise ValueError("comparison requires a supported dataset and a history limit of 20")
    sequence_hash = hashlib.sha256()
    for user in users:
        row = [user["user_id"], user["items"], user["timestamps"]]
        sequence_hash.update((json.dumps(row, separators=(",", ":")) + "\n").encode())
    catalog_ids = sorted(catalog, key=lambda item: item_key(dataset, item))
    return {"dataset": dataset, "max_history": max_history, "split_protocol": SPLIT_PROTOCOL,
            "sequences_sha256": sequence_hash.hexdigest(),
            "catalog_sha256": hashlib.sha256(json.dumps(catalog_ids).encode()).hexdigest()}


def sampled_users(users, limit):
    order = list(range(len(users)))
    random.Random(SEED).shuffle(order)
    return [users[i] for i in order[:limit]]


def train_position(user):
    return len(user["items"]) - HOLDOUT_OFFSETS["calibration"] - 1


def require_training_selection(record):
    if record.get("train_selection") != TRAIN_SELECTION:
        raise ValueError("Formal comparison requires one target per user before calibration. Retrain the model.")


def holdout(user, split):
    position = -HOLDOUT_OFFSETS[split]
    return user["items"][:position], user["items"][position]


def validation_users(users, *, smoke=False, pilot=False):
    if pilot:
        return sampled_users(users, PILOT_VALIDATION_USERS)
    return sampled_users(users, 8) if smoke else users


def validation_record(users, *, smoke=False):
    targets = [(user["user_id"], user["items"][-HOLDOUT_OFFSETS["validation"]]) for user in users]
    fingerprint = hashlib.sha256(json.dumps(targets, separators=(",", ":")).encode()).hexdigest()
    return {"scope": "smoke" if smoke else "full", "split": "validation",
            "metric": f"NDCG@{VALIDATION_K}", "users": len(users), "targets_sha256": fingerprint}


def require_full_validation(record, users):
    if record != validation_record(users):
        raise ValueError("Full evaluation requires a checkpoint selected on full validation NDCG@10. Retrain the model.")


def user_targets(users):
    return [{"user_id": user["user_id"], "target_item": user["items"][-1]} for user in users]


def select_cohort_users(users, entries, expected_count):
    if len(entries) != expected_count:
        raise ValueError(f"expected {expected_count} cohort entries, got {len(entries)}")
    users_by_id = {user["user_id"]: user for user in users}
    if len(users_by_id) != len(users):
        raise ValueError("sequences contain duplicate users")
    selected = []
    seen = set()
    for entry in entries:
        uid, target = entry["user_id"], entry["target_item"]
        if uid in seen or uid not in users_by_id:
            raise ValueError(f"unknown or duplicate cohort user: {uid}")
        user = users_by_id[uid]
        if target != user["items"][-1]:
            raise ValueError(f"cohort target mismatch for user {uid}")
        seen.add(uid)
        selected.append(user)
    return selected


def load_cohort(path, dataset, users, catalog, max_history):
    # Hash the same bytes that were parsed, including the ordered entries.
    content = path.read_bytes()
    cohort = json.loads(content)
    identity = dataset_identity(dataset, users, catalog, max_history)
    if cohort["identity"] != identity:
        raise ValueError(f"{path}: cohort dataset, sequences or catalog differ from the run")
    scope = cohort["scope"]
    if dataset == PILOT_DATASET:
        if scope != "pilot" or len(users) != PILOT_USERS or len(catalog) != PILOT_ITEMS:
            raise ValueError(f"{path}: expected the fixed MovieLens-100K pilot data and cohort")
        selected = select_cohort_users(users, cohort["entries"], PILOT_USERS)
        if cohort["entries"] != user_targets(users):
            raise ValueError(f"{path}: pilot test users differ from the prepared user order")
        return selected, {"dataset": dataset, "scope": scope, "top_k": TOP_K,
                          "cohort_sha256": hashlib.sha256(content).hexdigest()}
    if scope not in ("formal", "smoke"):
        raise ValueError(f"{path}: invalid cohort scope")
    expected = min(8, len(users)) if scope == "smoke" else COUNTS[dataset][0]
    selected = select_cohort_users(users, cohort["entries"], expected)
    if scope == "smoke" and cohort["entries"] != user_targets(sampled_users(users, expected)):
        raise ValueError(f"{path}: cohort does not match the fixed seed-{SEED} sample")
    return selected, {"dataset": dataset, "scope": scope, "top_k": TOP_K,
                      "cohort_sha256": hashlib.sha256(content).hexdigest()}


def ranking_metrics(ranks, cutoffs):
    if not ranks:
        raise ValueError("evaluation requires at least one user")
    metrics = {}
    for k in cutoffs:
        hits = [rank for rank in ranks if rank is not None and rank <= k]
        metrics[f"Hits@{k}"] = len(hits)
        metrics[f"Recall@{k}"] = len(hits) / len(ranks)
        metrics[f"NDCG@{k}"] = math.fsum(1 / math.log2(rank + 1) for rank in hits) / len(ranks)
    return metrics


def validation_metrics(entries, users, cutoff=VALIDATION_K):
    ranks = [entry["rank"] for entry in entries]
    if len(ranks) != len(users):
        raise ValueError("Validation did not return one rank per validation user")
    return {"users": len(ranks), **ranking_metrics(ranks, (cutoff,))}


def ranked_entry(user_id, target, recommendations):
    rank = next((i for i, row in enumerate(recommendations, 1) if row["item_id"] == target), None)
    return {"user_id": user_id, "target_item": target, "rank": rank, "recommendations": recommendations}


def _checked_entries(entries, selected, catalog):
    catalog = set(catalog)
    missing = object()
    for entry, user in zip_longest(entries, selected, fillvalue=missing):
        if entry is missing or user is missing:
            raise ValueError("result user count differs from the cohort")
        uid, target = user["user_id"], user["items"][-1]
        if (entry["user_id"], entry["target_item"]) != (uid, target):
            raise ValueError(f"result user or target differs from the cohort at {uid}")
        seen = set(user["items"][:-1])
        ids = [row["item_id"] for row in entry["recommendations"]]
        if (len(ids) != min(TOP_K, len(catalog) - len(seen)) or len(ids) != len(set(ids))
                or any(item in seen or item not in catalog for item in ids)):
            raise ValueError(f"invalid Top-{TOP_K} recommendations for user {uid}")
        rank = ids.index(target) + 1 if target in ids else None
        if entry["rank"] != rank:
            raise ValueError(f"invalid stored rank for user {uid}")
        yield entry


def checked_ranks(entries, selected, catalog):
    return [entry["rank"] for entry in _checked_entries(entries, selected, catalog)]


def add_comparison_metrics(result, selected, catalog, metadata, model):
    ranks = checked_ranks(result["entries"], selected, catalog)
    result.update(metadata, model=model, users=len(selected))
    result.update(ranking_metrics(ranks, METRIC_CUTOFFS))
    return result


def write_evaluation(output, entries, selected, catalog, metadata, model, **details):
    """Write and validate rankings incrementally, then publish a complete result."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".evaluation-{uuid4().hex}.tmp")
    ranks = []
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_EVALUATION_HEADER)
            for index, entry in enumerate(_checked_entries(entries, selected, catalog)):
                if index:
                    handle.write(",\n")
                handle.write(json.dumps(entry, allow_nan=False, separators=(",", ":")))
                ranks.append(entry["rank"])
            result = {**details, **metadata, "model": model, "users": len(ranks),
                      **ranking_metrics(ranks, METRIC_CUTOFFS)}
            handle.write("\n]," + json.dumps(result, allow_nan=False, indent=2)[1:] + "\n")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def read_evaluation(path, selected, catalog):
    """Read a streamed result in bounded memory, with support for existing JSON files."""
    with path.open(encoding="utf-8") as handle:
        if handle.readline() != _EVALUATION_HEADER:
            handle.seek(0)
            result = json.load(handle)
            return result, checked_ranks(result.pop("entries"), selected, catalog)

        def entries():
            for line in handle:
                if line.endswith(",\n"):
                    yield json.loads(line[:-2])
                else:
                    yield json.loads(line)
                    return

        ranks = checked_ranks(entries(), selected, catalog)
        footer = handle.readline()
        if not footer.startswith("],"):
            raise ValueError(f"{path}: missing evaluation summary")
        return json.loads("{" + footer[2:] + handle.read()), ranks
