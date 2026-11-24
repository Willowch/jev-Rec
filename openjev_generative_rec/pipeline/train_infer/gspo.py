"""GSPO sequence loss and positive NLL over complete SID trajectories."""
from dataclasses import dataclass

import torch

from ..data_preparation.build_jev_data import CatalogPolicy
from ..models.openjev_runtime import backward_logits, prepare_candidate_chunks, score


def group_advantages(rewards, eps=1e-8):
    rewards = rewards.detach()
    # 使用总体标准差；同奖励组的优势为零，仍可由 NLL 学习。
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def gspo_loss(log_probs, old_log_probs, advantages, clip_low=3e-4, clip_high=4e-4):
    """[group size, SID depth]; exponentiate the mean log-ratio before clipping."""
    if log_probs.ndim != 2 or log_probs.shape != old_log_probs.shape:
        raise ValueError("new/old log probabilities must have matching [group, tokens] shapes")
    if advantages.shape != log_probs.shape[:1] or not 0 < clip_low < 1 or clip_high <= 0:
        raise ValueError("invalid advantages or clipping range")
    ratio = ((log_probs - old_log_probs.detach()).mean(-1)).exp()
    advantage = advantages.detach()
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite GSPO loss")
    active_clip = ((advantage > 0) & (ratio > 1 + clip_high)) | ((advantage < 0) & (ratio < 1 - clip_low))
    return loss, {"ratio_mean": float(ratio.detach().mean()),
                  "clip_fraction": float(active_clip.float().mean())}


class PrefixScores:
    """缓存小型 logits 叶子；模型大图只在 backward_logits 时逐块重建。"""
    def __init__(self, model, history_ids, catalog, candidate_batch, initial=None, history_items=None, prepared=None):
        self.model, self.history_ids, self.catalog = model, history_ids, catalog
        self.candidate_batch, self.initial = candidate_batch, initial or {}
        self.rows, self.logits = {}, {}
        self.history_items = history_items
        self.prepared = prepared

    def distribution(self, prefix):
        prefix = tuple(prefix)
        if prefix not in self.logits:
            row = self.catalog.row(self.history_ids, prefix, self.history_items)
            values = self.initial.get(prefix)
            if values is None:
                inputs = None
                if self.prepared is not None:
                    if prefix not in self.prepared:
                        self.prepared[prefix] = prepare_candidate_chunks(self.model, row, self.candidate_batch)
                    inputs = self.prepared[prefix]
                values = score(self.model, row, self.candidate_batch, prepared=inputs)
            self.rows[prefix] = row
            self.logits[prefix] = values.detach().requires_grad_(True)
        return self.logits[prefix].log_softmax(-1)

    def trajectory(self, code):
        return torch.stack([self.distribution(code[:t])[self.catalog.options[code[:t]].index(token)]
                            for t, token in enumerate(code)])


@dataclass
class RolloutGroup:
    group_id: str
    history_ids: list
    target: tuple
    codes: list
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    initial_logits: dict
    history_items: list
    prepared_inputs: dict | None = None


@torch.no_grad()
def sample_group(model, row, item_ids, catalog, reward_fn, group_size=4, candidate_batch=1, *, reuse_encoding=False):
    if row["split"] != "train" or row["metadata"]["semantic_layer"] != 1 or group_size < 2:
        raise ValueError("rollouts require a training context at layer 1 and group_size >= 2")
    history = row["metadata"]["history_items"]
    history_ids = [item_ids[item] for item in history]
    cache = PrefixScores(model, history_ids, catalog, candidate_batch, history_items=history,
                         prepared={} if reuse_encoding else None)
    codes, old_log_probs = [], []
    for _ in range(group_size):
        prefix, values = (), []
        for _ in range(catalog.depth):
            log_probs = cache.distribution(prefix)
            action = int(torch.multinomial(log_probs.exp(), 1))
            values.append(log_probs[action])
            prefix += (catalog.options[prefix][action],)
        codes.append(prefix)
        old_log_probs.append(torch.stack(values))
    target = tuple(item_ids[row["metadata"]["target_item"]])
    old = torch.stack(old_log_probs).detach()
    values = (reward_fn.score_ranked(codes, target, old.sum(-1).tolist())
              if hasattr(reward_fn, "score_ranked") else reward_fn(codes, history, target))
    rewards = torch.as_tensor(values, device=old.device, dtype=old.dtype).detach()
    if rewards.shape != (group_size,) or not torch.isfinite(rewards).all():
        raise ValueError("reward function must return one finite reward per trajectory")
    return RolloutGroup(row["group_id"], history_ids, target, codes, old, rewards,
                        group_advantages(rewards), {p: v.detach() for p, v in cache.logits.items()},
                        list(history), cache.prepared)


def backward_gspo(model, group, catalog, candidate_batch=1, nll_weight=0.1,
                  clip_low=3e-4, clip_high=4e-4, scale=1.0, first_update=False):
    cache = PrefixScores(model, group.history_ids, catalog, candidate_batch,
                         group.initial_logits if first_update else None, history_items=group.history_items,
                         prepared=group.prepared_inputs)
    current = torch.stack([cache.trajectory(code) for code in group.codes])
    policy_loss, metrics = gspo_loss(current, group.old_log_probs, group.advantages, clip_low, clip_high)
    # The positive NLL sums log probabilities over the whole SID.
    nll = -cache.trajectory(group.target).sum() if nll_weight else current.new_zeros(())
    loss = policy_loss + nll_weight * nll
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite GSPO + NLL loss")
    policy_gradients = torch.autograd.grad(policy_loss, tuple(cache.logits.values()), retain_graph=True, allow_unused=True)
    policy_grad_norm = sum(g.square().sum() for g in policy_gradients if g is not None).sqrt()
    gradients = torch.autograd.grad(loss * scale, tuple(cache.logits.values()))
    for prefix, gradient in zip(cache.logits, gradients):
        inputs = cache.prepared.get(prefix) if cache.prepared is not None else None
        backward_logits(model, cache.rows[prefix], gradient, candidate_batch, prepared=inputs)
    metrics.update(loss=float(loss.detach()), gspo_loss=float(policy_loss.detach()), nll=float(nll.detach()),
                   policy_logit_grad_norm=float(policy_grad_norm),
                   reward_mean=float(group.rewards.mean()), reward_std=float(group.rewards.std(unbiased=False)),
                   zero_advantage_group=bool((group.advantages == 0).all()),
                   positive_code_rate=sum(code == group.target for code in group.codes) / len(group.codes))
    return metrics
