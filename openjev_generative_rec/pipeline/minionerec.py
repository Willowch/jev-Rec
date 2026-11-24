"""Selected MiniOneRec routines (Apache-2.0), with OpenJev data adapters.

Source: https://github.com/AkaliKong/MiniOneRec
See data_preparation/MiniOneRec-LICENSE.txt.
"""
import math
from collections import Counter, defaultdict
from functools import partial


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    """MiniOneRec sft.py."""
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    from torch.optim.lr_scheduler import LambdaLR

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda)


def ranked_group_rewards(matches):
    """MiniOneRec rl.py rule_reward + ndcg_rule_reward for one ranked group."""
    if not any(matches):
        return [0.0] * len(matches)
    weights = [-1.0 / math.log2(i + 2) for i in range(len(matches))]
    total = sum(weights)
    return [1.0 if match else -weight / total for match, weight in zip(matches, weights)]


def index_statistics(index):
    """MiniOneRec sid_generation/sid_common.py."""
    groups = Counter(tuple(tokens) for tokens in index.values())
    count = len(index)
    lengths = Counter(len(tokens) for tokens in index.values())
    return {
        "items": count,
        "unique_sids": len(groups),
        "excess_collision_count": count - len(groups),
        "excess_collision_rate": (count - len(groups)) / count if count else 0.0,
        "affected_item_count": sum(size for size in groups.values() if size > 1),
        "affected_item_rate": sum(size for size in groups.values() if size > 1) / count if count else 0.0,
        "max_collision_group": max(groups.values(), default=0),
        "length_histogram": {str(k): v for k, v in sorted(lengths.items())},
        "observed_vocabulary_size": len({token for tokens in index.values() for token in tokens}),
    }


def append_identity_tokens(raw):
    """MiniOneRec build_indexes suffix logic, preserving OpenJev's raw SID spelling."""
    raw_stats = index_statistics(raw)
    final = {key: list(tokens) for key, tokens in raw.items()}
    appended = raw_stats["excess_collision_count"] > 0
    if appended:
        groups = defaultdict(list)
        for item_id, tokens in raw.items():
            groups[tuple(tokens)].append(item_id)
        for members in groups.values():
            for rank, item_id in enumerate(sorted(members)):
                final[item_id].append(f"<d_{rank}>")
    return final, {"identity_suffix_appended": appended, "raw": raw_stats, "final": index_statistics(final)}
