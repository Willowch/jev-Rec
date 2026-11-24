"""Joint collaborative and OpenJev semantic item scoring."""

import torch
from torch import nn
from torch.nn import functional as F


class FullCatalogHead(nn.Module):
    def __init__(self, item_features, history_limit, *, width=128):
        super().__init__()
        self.register_buffer("item_features", item_features, persistent=False)
        self.pad = len(item_features)
        self.item_embedding = nn.Embedding(self.pad + 1, width, padding_idx=self.pad)
        self.position_embedding = nn.Embedding(history_limit, width)
        block = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=2 * width, dropout=0.2, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(block, num_layers=2, enable_nested_tensor=False)
        self.sequence_norm = nn.LayerNorm(width)
        feature_width = item_features.shape[1]
        self.query = nn.Sequential(nn.LayerNorm(feature_width), nn.Linear(feature_width, width),
                                   nn.GELU(), nn.Dropout(0.2), nn.Linear(width, width))
        self.item_projection = nn.Linear(feature_width, width, bias=False)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[self.pad].zero_()

    def forward(self, history, context):
        padding = history.eq(self.pad)
        positions = torch.arange(history.shape[1], device=history.device)
        embedded = self.item_embedding(history) + self.position_embedding(positions)
        encoded = self.sequence(embedded, src_key_padding_mask=padding)
        last = padding.logical_not().sum(dim=1) - 1
        summary = self.sequence_norm(encoded[torch.arange(len(history), device=history.device), last])
        items = F.normalize(self.item_embedding.weight[:self.pad], dim=-1)
        collaborative = 10 * (F.normalize(summary, dim=-1) @ items.T)
        semantic_items = F.normalize(self.item_projection(self.item_features), dim=-1)
        semantic = 10 * (F.normalize(self.query(context), dim=-1) @ semantic_items.T)
        return collaborative, semantic


def standardize(scores):
    return (scores - scores.mean(dim=-1, keepdim=True)) / scores.std(
        dim=-1, correction=0, keepdim=True,
    ).clamp_min(1e-6)


def catalog_scores(collaborative, semantic, popularity, semantic_weight, popularity_weight):
    return collaborative + semantic_weight * semantic + popularity_weight * popularity


def target_ranks(scores, targets, seen):
    """Use numeric catalog order for ties and treat a filtered target as unranked."""
    scores = scores.masked_fill(seen, -torch.inf)
    target_scores = scores.gather(1, targets[:, None])
    order = torch.arange(scores.shape[-1], device=scores.device)
    ahead = (scores > target_scores) | ((scores == target_scores) & (order < targets[:, None]))
    ranks = ahead.sum(dim=1) + 1
    return ranks.masked_fill(seen.gather(1, targets[:, None]).squeeze(1), scores.shape[1] + 1)
