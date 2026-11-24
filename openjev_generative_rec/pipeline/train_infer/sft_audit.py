"""Compare a short SFT run with the full-catalog single-GPU reference."""
import argparse
import json
import math
from pathlib import Path

import torch


def record_choice(records, row, logits):
    records.append({"id": row["id"], "options": row["options"], "target": row["target"],
                    "logits": logits.detach().cpu()})


def record_gradients(trainable):
    groups = {"lora": [], "head": [], "sid_embedding": []}
    active = []
    for name, parameter in trainable.items():
        if parameter.grad is None:
            continue
        if name.startswith("head."):
            group = "head"
        elif name.endswith(".sid_weight"):
            group = "sid_embedding"
        else:
            group = "lora"
        groups[group].append(parameter.grad.detach().flatten().float().cpu())
        active.append(name)
    return {"active": active, "groups": {name: torch.cat(parts) if parts else torch.empty(0)
                                         for name, parts in groups.items()}}


def compare(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError("Audits must cover the same nonempty step range")
    worst = {"max_logit_abs_error": 0.0, "max_loss_abs_error": 0.0,
             "max_gradient_relative_l2": {"lora": 0.0, "head": 0.0, "sid_embedding": 0.0}}
    failures = []
    for expected, actual in zip(reference, candidate):
        step = expected["step"]
        for key in ("step", "sample_ids", "learning_rates"):
            if expected[key] != actual[key]:
                raise ValueError(f"Step {step}: {key} differs")
        loss_error = abs(expected["loss"] - actual["loss"])
        worst["max_loss_abs_error"] = max(worst["max_loss_abs_error"], loss_error)
        if not math.isfinite(loss_error) or loss_error > 0.02:
            failures.append(f"Step {step}: loss error {loss_error:.6g}")
        for left, right in zip(expected["choices"], actual["choices"], strict=True):
            for key in ("id", "options", "target"):
                if left[key] != right[key]:
                    raise ValueError(f"Step {step}: choice {key} differs")
            error = float((left["logits"] - right["logits"]).abs().max())
            worst["max_logit_abs_error"] = max(worst["max_logit_abs_error"], error)
            if not math.isfinite(error) or error > 0.05:
                failures.append(f"Step {step}: logits error {error:.6g} for {left['id']}")
        left, right = expected["gradients"], actual["gradients"]
        if left["active"] != right["active"]:
            raise ValueError(f"Step {step}: trainable parameters receiving gradients differ")
        for group, gradient in left["groups"].items():
            delta = float((gradient - right["groups"][group]).norm())
            norm = float(gradient.norm())
            worst["max_gradient_relative_l2"][group] = max(worst["max_gradient_relative_l2"][group],
                                                            delta / max(norm, 1e-5))
            if not math.isfinite(delta) or not math.isfinite(norm) or delta > 1e-5 + 0.02 * norm:
                failures.append(f"Step {step}: {group} gradient L2 difference {delta:.6g} (reference {norm:.6g})")
    return {"passed": not failures, "steps": len(reference), **worst, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    report = compare(reference, candidate)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
