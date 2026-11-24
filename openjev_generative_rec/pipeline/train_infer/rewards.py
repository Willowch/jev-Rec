"""本地可复现的 embedding 奖励替代物；不是 GenRec 的 SIM 偏好模型。"""
import json
from pathlib import Path

import numpy as np

from ..common import sha256
from ..minionerec import ranked_group_rewards


class MiniOneRecRankedReward:
    metadata = {"kind": "minionerec_hit_plus_rank_penalty",
                "ordering": "descending old-policy sequence log probability; sample order breaks ties",
                "positive_policy": "full SID match", "no_hit_group": "zero"}

    def score_ranked(self, codes, target, sequence_log_probs):
        order = sorted(range(len(codes)), key=lambda i: -sequence_log_probs[i])
        ranked = ranked_group_rewards([codes[i] == target for i in order])
        rewards = [0.0] * len(codes)
        for index, value in zip(order, ranked, strict=True):
            rewards[index] = value
        return rewards


class EmbeddingReward:
    def __init__(self, directory, catalog, threshold=0.0):
        directory = Path(directory)
        with (directory / "items.jsonl").open(encoding="utf-8") as handle:
            items = [row["item_id"] for row in map(json.loads, handle)]
        vectors = np.load(directory / "embeddings.npy").astype(np.float32)
        if vectors.ndim != 2 or len(vectors) != len(items) or len(set(items)) != len(items) or not np.isfinite(vectors).all():
            raise ValueError("invalid embedding matrix or item mapping")
        catalog_items = {item for group in catalog.items_by_code.values() for item in group}
        if catalog_items != set(items):
            raise ValueError("reward embeddings and SID catalog must contain the same items")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 1e-12).any() or not -1 <= threshold <= 1:
            raise ValueError("zero-norm embeddings or invalid cosine threshold")
        self.vectors = dict(zip(items, vectors / norms))
        self.catalog, self.threshold = catalog, threshold
        self.metadata = {"kind": "frozen_embedding_history_cosine_surrogate", "threshold": threshold,
                         "embeddings_sha256": sha256(directory / "embeddings.npy"),
                         "items_sha256": sha256(directory / "items.jsonl"),
                         "collision_policy": "mean reward over items sharing the full SID",
                         "positive_policy": "training-target SID match gets group maximum"}

    def __call__(self, codes, history, target):
        profile = np.mean([self.vectors[item] for item in history], axis=0)
        profile = profile / max(float(np.linalg.norm(profile)), 1e-12)
        seen = set(history)
        rewards = []
        for code in codes:
            values = []
            for item in self.catalog.items_by_code.get(code, []):
                similarity = float(np.clip(self.vectors[item] @ profile, -1, 1))
                values.append((similarity + 1) / 2 if item not in seen and similarity > self.threshold else 0.0)
            # SID 碰撞无法解歧义，平均所有成员的奖励，避免择优虚增分数。
            rewards.append(float(np.mean(values)) if values else 0.0)
        maximum = max(rewards)
        # 只用当前训练标签；此处的 positive 是语义码命中，不声称精确物品命中。
        return [maximum if code == target else value for code, value in zip(codes, rewards)]
