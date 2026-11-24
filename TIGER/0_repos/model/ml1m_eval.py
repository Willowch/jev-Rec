"""Movie-level evaluation using the OpenJev SFT ranking protocol."""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path


PROTOCOL = "full catalog, seen items excluded, autoregressive beam, no teacher forcing"


def comparison_protocol():
    # TIGER also supports standalone runs without the sibling comparison package.
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from comparison import protocol

    return protocol


def select_users(users, seed, limit):
    indices = list(range(len(users)))
    random.Random(seed).shuffle(indices)
    return [users[index] for index in (indices[:limit] if limit else indices)]


def recommend(item_codes, history, score_prefixes, *, layers, beam_size, top_k, temperature):
    """Score all tokens of a layer, then expand only prefixes of unseen catalog items."""
    seen = set(history)
    allowed = defaultdict(set)
    terminal_items = defaultdict(list)
    for item, code in item_codes.items():
        if item in seen:
            continue
        terminal_items[code].append(item)
        for layer in range(layers):
            allowed[code[:layer]].add(code[layer])

    if not terminal_items:
        return []
    beams = [((), 0.0)]
    for _ in range(layers):
        logits = score_prefixes([prefix for prefix, _ in beams]).float()
        log_probs = (logits / temperature).log_softmax(-1).cpu().tolist()
        expansions = [
            (prefix + (token,), accumulated + scores[token])
            for (prefix, accumulated), scores in zip(beams, log_probs)
            for token in allowed[prefix]
        ]
        beams = sorted(expansions, key=lambda pair: (-pair[1], pair[0]))[:beam_size]

    matches = [
        {"item_id": item, "semantic_ids": [f"sid_{digit:03d}" for digit in code],
         "log_probability": log_probability}
        for code, log_probability in beams
        for item in terminal_items[code]
    ]
    return sorted(matches, key=lambda row: (-row["log_probability"], int(row["item_id"])))[:top_k]


def evaluate_cases(cases, predict, *, beam_size, top_k, seed):
    entries = []
    for user_id, history, target in cases:
        recommendations = predict(history)
        ids = [row["item_id"] for row in recommendations]
        entries.append({"user_id": user_id, "target_item": target,
                        "rank": ids.index(target) + 1 if target in ids else None,
                        "recommendations": recommendations})

    count = len(entries)
    if not count:
        raise ValueError("Evaluation requires at least one user")
    ranks = [row["rank"] for row in entries if row["rank"] is not None]
    return {"users": count, f"Recall@{top_k}": len(ranks) / count,
            f"NDCG@{top_k}": sum(1 / math.log2(rank + 1) for rank in ranks) / count,
            "protocol": PROTOCOL, "policy_normalization": "full", "beam_size": beam_size,
            "seed": seed, "entries": entries}


def fit_temperature(logits, targets):
    """OpenJev jev.metrics.fit_temperature for hard targets on TIGER's Python 3.9."""
    if not logits or len(logits) != len(targets):
        raise ValueError("Calibration logits and targets must have equal nonzero length")
    prepared = []
    for row, target in zip(logits, targets):
        if not 0 <= target < len(row) or not all(map(math.isfinite, row)):
            raise ValueError("Invalid calibration logits or target")
        maximum = max(row)
        shifted = [value - maximum for value in row]
        prepared.append((shifted, shifted[target]))

    def loss(log_temperature):
        inverse = math.exp(-log_temperature)
        return sum(
            math.log(sum(math.exp(value * inverse) for value in row)) - expected * inverse
            for row, expected in prepared
        ) / len(prepared)

    lower, upper = math.log(0.05), math.log(20.0)
    grid = sorted({0.0, *(lower + index * (upper - lower) / 24 for index in range(25))})
    losses = [loss(point) for point in grid]
    best = min(range(len(grid)), key=lambda index: (losses[index], abs(grid[index])))
    candidates = [(losses[best], grid[best])]
    left, right = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    ratio = (math.sqrt(5) - 1) / 2
    first, second = right - ratio * (right - left), left + ratio * (right - left)
    first_loss, second_loss = loss(first), loss(second)
    for _ in range(24):
        if first_loss < second_loss:
            right, second, second_loss = second, first, first_loss
            first = right - ratio * (right - left)
            first_loss = loss(first)
        else:
            left, first, first_loss = first, second, second_loss
            second = left + ratio * (right - left)
            second_loss = loss(second)
    candidates.extend([(first_loss, first), (second_loss, second)])
    return math.exp(min(candidates, key=lambda candidate: (candidate[0], abs(candidate[1])))[1])
