"""从已发布 OpenJev LoRA/head 继续训练；不触碰测试集。"""
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch

from ..common import ROOT, enable_comparison, sha256
from ..data_preparation.build_jev_data import SEMANTIC_LAYERS, TRAIN_SELECTION, SequenceData
from ..minionerec import get_cosine_schedule_with_warmup
from ..models.openjev_runtime import backward_choice, load_model, measure_phase, score, set_training_mode
from .candidate_parallel import CandidateParallel
from .gspo import backward_gspo, sample_group
from .recommend import recommend
from .rewards import EmbeddingReward, MiniOneRecRankedReward
from .sft_audit import record_choice, record_gradients


def upstream_revision(upstream):
    if (upstream / ".git").exists():
        try:
            revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                               stderr=subprocess.DEVNULL, text=True).strip()
            if re.fullmatch(r"[0-9a-f]{40}", revision):
                return revision, "git"
        except (OSError, subprocess.CalledProcessError):
            pass
    packaged_revision = upstream / "UPSTREAM_COMMIT"
    if packaged_revision.is_file():
        revision = packaged_revision.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            return revision, "UPSTREAM_COMMIT"
    raise ValueError(f"No valid OpenJev Git HEAD or packaged revision in {upstream}")


def sft_rows(data, position, levels, catalog):
    rows = (data.row("train", position * data.depth + level, catalog=catalog) for level in levels)
    return [row for row in rows if len(row["options"]) > 1]


def shuffled_positions(size, rng):
    if size < 1:
        raise ValueError("At least one training position is required")
    order = list(range(size))
    while True:
        rng.shuffle(order)
        yield from order


@torch.inference_mode()
def validation_ndcg(model, data, users, candidate_batch, scoring_backend="batched"):
    from comparison.protocol import PILOT_DATASET, TOP_K, VALIDATION_K, item_key, ranking_metrics

    dataset = data.manifest.get("dataset", "ml1m")
    beam_size = VALIDATION_K if dataset == PILOT_DATASET else TOP_K
    ranks = []
    try:
        model.eval()
        for index, user in enumerate(users, 1):
            history, target = user["items"][:-2], user["items"][-2]
            results = recommend(model, data.item_ids, history, max_history=data.manifest["max_history"],
                                beam_size=beam_size, top_k=VALIDATION_K, candidate_batch=candidate_batch,
                                normalization="catalog", batch_prefixes=True,
                                catalog_key=lambda item: item_key(dataset, item), catalog_policy=data.catalog,
                                scoring_backend=scoring_backend, skip_singletons=dataset == PILOT_DATASET)
            ranks.append(next((rank for rank, row in enumerate(results, 1) if row["item_id"] == target), None))
            if dataset == PILOT_DATASET and index % 16 == 0:
                print(json.dumps({"event": "validation_progress", "done": index, "total": len(users)}), flush=True)
    finally:
        set_training_mode(model, True)
    return ranking_metrics(ranks, (10,))["NDCG@10"]


def save_calibrated_checkpoint(model, data, args, checkpoint, normalization, sid_encoding):
    from jev.metrics import fit_temperature

    model.save(checkpoint)
    gspo = args.objective == "gspo"
    if gspo:
        config = json.loads((checkpoint / "model.json").read_text())
        config["method"] = ("independent_candidate_lora_sid_embedding_gspo_nll" if hasattr(model, "sid_manifest")
                            else "independent_candidate_lora_gspo_nll")
        (checkpoint / "model.json").write_text(json.dumps(config, indent=2))

    # Use only the dedicated calibration split and evaluate each candidate in inference mode.
    cal_rng = random.Random(args.calibration_seed if args.calibration_seed is not None else args.seed + 1)
    count = data.count("calibration")
    indices = cal_rng.sample(range(count), min(args.calibration_rows, count))
    sparse = args.sft_recipe == "minionerec" or (gspo and (args.reward != "embedding" or data.depth > SEMANTIC_LAYERS))
    values, targets = [], []
    training = model.training
    model.eval()
    try:
        for index in indices:
            row = data.row("calibration", index, catalog=sparse)
            logits = score(model, row, args.candidate_batch).cpu()
            if not values:
                first_row, reference = row, logits.clone()
            if gspo and not sparse:
                prefix = tuple(data.item_ids[row["metadata"]["target_item"]][:row["metadata"]["semantic_layer"] - 1])
                allowed = [row["options"].index(token) for token in data.catalog.options[prefix]]
                values.append(logits[allowed].tolist())
                targets.append([row["target"][i] for i in allowed])
            else:
                values.append(logits.tolist())
                targets.append(row["target"])
    finally:
        set_training_mode(model, training)
    temperature = fit_temperature(values, targets)
    (checkpoint / "temperature.json").write_text(json.dumps({
        "temperature": temperature, "split": "calibration", "indices": indices, "normalization": normalization,
    }, indent=2))
    (checkpoint / "recommendation.json").write_text(json.dumps({
        "max_history": data.manifest["max_history"], "sid_encoding": sid_encoding,
        "policy_normalization": normalization, "semantic_ids_sha256": data.manifest["semantic_ids_sha256"],
    }, indent=2))
    return temperature, first_row, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ml1m_sequences"))
    parser.add_argument("--comparison-data", type=Path, help="Shared sequence directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/openjev-2b/package/checkpoint"))
    parser.add_argument("--base-model", default="models/qwen3.5-2b")
    parser.add_argument("--output", type=Path, default=Path("runs/ml1m-lora"))
    parser.add_argument("--steps", type=int, help="Default: 300 for MiniOneRec SFT, otherwise 1000")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--candidate-batch", type=int, default=1)
    parser.add_argument("--prefix-candidate-batch", type=int,
                        help="Candidate batch size after a shared prefix; defaults to --candidate-batch")
    parser.add_argument("--training-scorer", choices=("legacy", "prepared", "prefix", "prefix-all"), default="legacy",
                        help="Share model states within the first SID layer (prefix) or across SID layers (prefix-all)")
    parser.add_argument("--gradient-checkpointing", choices=("on", "off", "partial"), default="on",
                        help="partial keeps activation checkpointing on three of every four decoder blocks")
    parser.add_argument("--candidate-parallel", action="store_true",
                        help="Shard each full SFT choice across workers launched with single-node torchrun")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Collect synchronized phase timings for the first N prepared or prefix SFT steps")
    parser.add_argument("--audit-sft-steps", type=int, default=0,
                        help="Save logits and pre-clipping gradients for the first N SFT steps")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--sid-lr", type=float, default=1e-4)
    parser.add_argument("--sid-encoding", choices=("auto", "layered", "text"), default="auto",
                        help="auto: 新 SFT 启用分层 SID，已有 checkpoint 按格式继续；text: 复现旧基线")
    parser.add_argument("--history-memory", choices=("auto", "on", "off"), default="auto",
                        help="auto: 分层 SFT 启用历史文字记忆，GSPO 恢复 checkpoint 配置；off: 无记忆对照")
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--objective", choices=("sft", "gspo"), default="sft")
    parser.add_argument("--full-validation", action="store_true",
                        help="Save the source SFT, midpoint GSPO and final GSPO for full validation")
    parser.add_argument("--sft-recipe", choices=("legacy", "minionerec"), default="legacy")
    parser.add_argument("--sft-all-layers", action="store_true", help="Train all informative SID layers per sampled user")
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-users", type=int, default=8, help="Use 0 to keep the SFT weights from the final step")
    parser.add_argument("--validation-backend", choices=("batched", "prefix-cache"), default="batched")
    parser.add_argument("--validation-candidate-batch", type=int, help="Defaults to --candidate-batch")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rollout-updates", type=int, default=2, help="同一组旧策略采样复用的优化步数")
    parser.add_argument("--clip-low", type=float, default=3e-4)
    parser.add_argument("--clip-high", type=float, default=4e-4)
    parser.add_argument("--nll-weight", type=float, default=0.1, help="GSPO 正样本整条轨迹 NLL 的权重")
    parser.add_argument("--reward-embeddings", type=Path, default=Path("data/ml1m_qwen_embeddings"))
    parser.add_argument("--reward-threshold", type=float, default=0.0)
    parser.add_argument("--reward", choices=("embedding", "minionerec-ranked"), default="embedding")
    parser.add_argument("--calibration-rows", type=int, default=30)
    parser.add_argument("--calibration-seed", type=int, help="Defaults to the training seed plus one")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    gspo = args.objective == "gspo"
    new_sft = args.sft_recipe == "minionerec"
    prefix_scoring = args.training_scorer in ("prefix", "prefix-all")
    if (gspo and new_sft) or (not gspo and args.reward != "embedding"):
        parser.error("--sft-recipe selects an SFT recipe and --reward selects a GSPO reward")
    if args.sft_all_layers and not new_sft:
        parser.error("--sft-all-layers requires --sft-recipe minionerec")
    if args.steps is None:
        args.steps = 300 if new_sft else 1000
    if args.full_validation and (not gspo or args.comparison_data is None or args.steps < 2):
        parser.error("--full-validation requires GSPO, --comparison-data and at least two steps")
    if min(args.steps, args.accumulation, args.candidate_batch, args.calibration_rows) < 1:
        raise ValueError("steps, accumulation, candidate-batch and calibration-rows must be positive")
    if args.validation_candidate_batch is not None and args.validation_candidate_batch < 1:
        parser.error("--validation-candidate-batch must be positive")
    if gspo and (args.group_size < 2 or args.rollout_updates < 1 or
            not 0 < args.clip_low < 1 or not 0 < args.clip_high or not 0 <= args.nll_weight):
        raise ValueError("invalid GSPO group size, update count, clipping range or NLL weight")
    if new_sft and (args.validation_every < 1 or min(args.validation_users, args.warmup_steps) < 0):
        parser.error("Validation interval must be positive; user count and warmup must be nonnegative")
    if not 0 <= args.profile_steps <= args.steps or not 0 <= args.audit_sft_steps <= args.steps:
        parser.error("Profile and audit step counts must be between zero and --steps")
    if args.profile_steps and (gspo or args.training_scorer == "legacy"):
        parser.error("--profile-steps requires prepared or prefix SFT")
    if prefix_scoring and (gspo or not new_sft or not args.sft_all_layers
                           or args.sid_encoding == "text" or args.candidate_parallel):
        parser.error("Prefix scoring requires single-GPU layered MiniOneRec SFT with --sft-all-layers")
    if args.prefix_candidate_batch is not None and (not prefix_scoring or args.prefix_candidate_batch < 1):
        parser.error("--prefix-candidate-batch must be positive and requires prefix scoring")
    if prefix_scoring and args.prefix_candidate_batch is None:
        args.prefix_candidate_batch = args.candidate_batch
    if gspo and args.audit_sft_steps:
        parser.error("--audit-sft-steps requires SFT")
    if args.candidate_parallel and (gspo or not new_sft or not args.sft_all_layers or args.validation_users
                                    or args.training_scorer != "prepared"):
        parser.error("Candidate parallel requires prepared MiniOneRec SFT, --sft-all-layers and --validation-users 0")
    if not args.candidate_parallel and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Multi-worker training requires --candidate-parallel")

    parallel = CandidateParallel.from_environment() if args.candidate_parallel else None
    try:
        train(args, parallel)
    finally:
        if parallel is not None:
            parallel.close()


def train(args, parallel):
    startup_started = time.time()
    gspo = args.objective == "gspo"
    new_sft = args.sft_recipe == "minionerec"
    reuse_encoding = args.training_scorer == "prepared"
    primary = parallel is None or parallel.rank == 0
    if primary:
        if (args.output / "training.jsonl").exists() or (args.output / "checkpoint").exists():
            raise FileExistsError("output already used; select a new --output")
        args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = SequenceData(args.data, args.comparison_data)
    if not data.train_positions or not data.users:
        raise ValueError("Training requires at least one training position")
    if data.depth > SEMANTIC_LAYERS and not (gspo or new_sft):
        raise ValueError("Identity SID SFT requires --sft-recipe minionerec")
    if sha256(args.data / "item_semantic_ids.jsonl") != data.manifest["semantic_ids_sha256"]:
        raise ValueError("OpenJev SID catalog differs from the data manifest")
    revision, revision_source = upstream_revision(ROOT / "third_party/Open-Jev")
    if args.full_validation:
        from .gspo_selection import candidate_fingerprints, candidate_path, candidate_steps, require_sft_source
        from comparison.protocol import staged_output

        require_sft_source(args.checkpoint, data)
        midpoint, _ = candidate_steps(args.steps)
    if args.objective == "gspo" and args.sid_encoding != "text" and not (args.checkpoint / "sid_tokens.json").exists():
        raise ValueError("请先用分层 SID 完成 SFT，再以该 checkpoint 进行 GSPO；复现旧版本请用 --sid-encoding text")
    catalog = data.catalog
    normalization = "catalog" if new_sft or gspo else "full"
    reward = None
    if gspo:
        reward = (MiniOneRecRankedReward() if args.reward == "minionerec-ranked"
                  else EmbeddingReward(args.reward_embeddings, catalog, args.reward_threshold))
    previous_mapping = args.checkpoint / "recommendation.json"
    if previous_mapping.exists():
        saved = json.loads(previous_mapping.read_text())
        if saved["semantic_ids_sha256"] != data.manifest["semantic_ids_sha256"]:
            raise ValueError("继续微调时语义 ID 映射必须与已有推荐 checkpoint 一致")
    source_config = json.loads((args.checkpoint / "model.json").read_text())
    memory_mode = args.history_memory
    if memory_mode == "auto" and args.objective == "sft" and args.sid_encoding != "text":
        memory_mode = "on"
    if args.objective == "gspo" and memory_mode == "on" and not source_config.get("history_memory"):
        raise ValueError("请先用历史文字记忆完成 SFT，再以该 checkpoint 进行 GSPO")
    checkpointing = {"on": True, "off": False, "partial": "partial"}[args.gradient_checkpointing]
    model = load_model(args.checkpoint, args.base_model, train=True, sid_encoding=args.sid_encoding,
                       history_memory=memory_mode, item_catalog=args.data / "item_semantic_ids.jsonl",
                       device=parallel.device if parallel is not None else "cuda:0",
                       gradient_checkpointing=checkpointing)
    prefix_usage = None
    choice_group_backward = None
    if args.training_scorer in ("prefix", "prefix-all"):
        from ..models.training_prefix import backward_prefix_all, backward_prefix_choice

        if model.backbone.config.model_type != "qwen3_5_text" or not hasattr(model, "sid_manifest") or model.sharded:
            raise ValueError("Prefix scoring requires one resident Qwen3.5 text model with layered SID embeddings")
        prefix_usage = {"choices": 0, "prefills": 0, "forward_tokens_saved": 0, "choices_by_layer": {}}
        if args.training_scorer == "prefix-all":
            choice_group_backward = partial(backward_prefix_all, prefix_batch_size=args.prefix_candidate_batch,
                                            usage=prefix_usage)
    unexpected = [name for name, parameter in model.named_parameters() if parameter.requires_grad
                  and not (name.startswith("head.") or ".lora_A." in name or ".lora_B." in name
                           or name.endswith(".sid_weight"))]
    if unexpected:
        raise ValueError(f"unexpected trainable parameters: {unexpected}")
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    parameters = list(trainable.values())
    if parallel is not None:
        parallel.synchronize_parameters(parameters)
    initial_parameters = ({name: p.detach().cpu().clone() for name, p in trainable.items()}
                          if primary and (gspo or hasattr(model, "sid_manifest")) else {})
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.backbone.named_parameters()
                    if p.requires_grad and not n.endswith(".sid_weight")], "lr": args.lr},
        {"params": [p for n, p in model.backbone.named_parameters()
                    if p.requires_grad and n.endswith(".sid_weight")], "lr": args.sid_lr},
        {"params": list(model.head.parameters()), "lr": args.head_lr}], weight_decay=0.01)
    best_validation = None
    if new_sft:
        enable_comparison()
        from comparison.protocol import (PILOT_DATASET, VALIDATION_K, sampled_users, validation_record,
                                         validation_users as shared_validation_users)

        pilot = data.manifest.get("dataset") == PILOT_DATASET
        validation_users = (shared_validation_users(data.users, pilot=True) if pilot and args.validation_users
                            else sampled_users(data.users, args.validation_users))
        positions = shuffled_positions(data.train_positions, rng)
        seen_positions = set()
        warmup_steps = min(args.warmup_steps, args.steps - 1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.steps)
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "data_manifest": data.manifest, "source_checkpoint_sha256": sha256(args.checkpoint / "head.pt"),
                "comparison_identity": data.comparison_identity,
                "train_selection": TRAIN_SELECTION, "train_positions": data.train_positions,
                "train_sid_decisions": data.train_decisions,
                "openjev_commit": revision, "openjev_commit_source": revision_source,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "gradient_checkpointing": args.gradient_checkpointing,
                "candidate_parallel_world_size": parallel.world_size if parallel is not None else 1,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "linear_attention_kernels": sorted({module.chunk_gated_delta_rule.__name__
                    for module in model.modules() if hasattr(module, "chunk_gated_delta_rule")}),
                "causal_conv1d_kernels": sorted({getattr(module.causal_conv1d_fn, "__name__", "torch_conv1d")
                    for module in model.modules() if hasattr(module, "causal_conv1d_fn")}),
                "sampling": "uniform training positions; cyclic semantic layers; with replacement",
                "max_history": data.manifest["max_history"], "status": "running"}
    metadata["sid_encoding"] = getattr(model, "sid_manifest", {}).get("format", "text")
    metadata["sid_embedding_parameters"] = sum(p.numel() for n, p in model.named_parameters() if n.endswith(".sid_weight"))
    if os.environ.get("OPENJEV_CPU_TOKEN_EMBEDDINGS") == "1":
        metadata["base_token_embedding_device"] = str(model.backbone.get_input_embeddings().weight.device)
    if os.environ.get("JEV_LOAD_4BIT") == "1":
        metadata["backbone_quantization"] = "nf4"
    if prefix_usage is not None and os.environ.get("OPENJEV_RECOMPUTE_PREFIX") == "1":
        metadata["recompute_prefix"] = True
    memory = getattr(model, "history_memory", None)
    metadata["history_memory"] = "history_text_v1" if memory is not None else "off"
    if new_sft:
        metadata.update(sampling="shuffled training positions without replacement",
                        sid_training="all informative layers" if args.sft_all_layers else "cyclic SID layers",
                        policy_normalization=normalization, warmup_steps=warmup_steps,
                        validation_users=[row["user_id"] for row in validation_users])
        if validation_users:
            metadata.update(validation_metric="NDCG@10", validation_split="penultimate",
                            validation_beam_size=VALIDATION_K if pilot else 20)
            if pilot:
                metadata["validation_protocol"] = validation_record(validation_users)
    if primary and memory is not None:
        from jev.api import candidate_prompts
        # 保存模型实际可见的 prompt，便于检查文字记忆；不包含目标标签。
        example = model.prepare_record(data.row("train", min(57, data.train_decisions - 1)))
        prompt = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": candidate_prompts(example)[0]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        metadata["example_prompt_tokens"] = len(model.tokenizer(prompt)["input_ids"])
        if metadata["example_prompt_tokens"] > model.max_length:
            raise ValueError("history memory prompt exceeds model max_length; reduce max_history explicitly")
        (args.output / "prompt_example.txt").write_text(prompt, encoding="utf-8")
    if gspo:
        metadata.update(sampling="uniform training positions; autoregressive catalog-masked multinomial groups; temperature=1",
                        policy_normalization="catalog", reward=reward.metadata,
                        loss="GSPO sequence clipping + nll_weight * sum(positive token NLL); no Brier or KL",
                        references=["https://arxiv.org/abs/2507.18071", "https://arxiv.org/abs/2604.14878"])
    if args.full_validation:
        with staged_output(candidate_path(args.output, 0)) as stage:
            shutil.copytree(args.checkpoint, stage, dirs_exist_ok=True)
        metadata["includes_sft_candidate"] = True
    if primary:
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"event": "loaded", "trainable_parameters": metadata["trainable_parameters"]}), flush=True)
    if parallel is not None:
        torch.distributed.barrier()
    started = time.time()
    if parallel is not None:
        choice_backward = parallel.backward_choice
    elif args.training_scorer == "prefix":
        choice_backward = partial(backward_prefix_choice, prefix_batch_size=args.prefix_candidate_batch,
                                  usage=prefix_usage)
    else:
        choice_backward = partial(backward_choice, reuse_encoding=reuse_encoding)
    audit_steps = []
    effective_decisions = 0
    sampled_contexts = hit_groups = nonzero_advantage_groups = 0
    sampled_context_ids = set()
    steps_with_policy_gradient = 0
    with ((args.output / "training.jsonl").open("w") if primary else nullcontext()) as log:
        for step in range(args.steps):
            timings = {} if step < args.profile_steps else None
            audit_records = [] if primary and step < args.audit_sft_steps else None
            audit_choice = partial(record_choice, audit_records) if audit_records is not None else None
            new_rollout = gspo and step % args.rollout_updates == 0
            if new_rollout:
                # 整个 accumulation 批次先由同一个 old policy 采样，之后才做参数更新。
                groups = []
                for _ in range(args.accumulation):
                    row = data.row("train", data.depth * rng.randrange(data.train_positions), catalog=True)
                    groups.append(sample_group(model, row, data.item_ids, catalog, reward,
                                               args.group_size, args.candidate_batch, reuse_encoding=reuse_encoding))
                print(json.dumps({"event": "rollout", "step": step + 1,
                                  "groups": [g.group_id for g in groups]}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            loss_sum, sample_ids, group_metrics = 0, [], []
            step_decisions = 0
            for micro in range(args.accumulation):
                if gspo:
                    group = groups[micro]
                    metrics = backward_gspo(model, group, catalog, args.candidate_batch, args.nll_weight,
                                            args.clip_low, args.clip_high, 1 / args.accumulation,
                                            first_update=new_rollout)
                    loss_sum += metrics["loss"] / args.accumulation
                    sample_ids.append(group.group_id)
                    group_metrics.append({**metrics, "codes": group.codes, "target_code": group.target,
                                          "rewards": group.rewards.cpu().tolist(),
                                          "advantages": group.advantages.cpu().tolist(),
                                          "old_log_probs": group.old_log_probs.cpu().tolist()})
                else:
                    level = (step * args.accumulation + micro) % data.depth
                    position = next(positions) if new_sft else rng.randrange(data.train_positions)
                    levels = range(data.depth) if args.sft_all_layers else (level,)
                    rows = sft_rows(data, position, levels, new_sft)
                    if new_sft and rows:
                        seen_positions.add(position)
                    step_decisions += len(rows)
                    if not rows:
                        continue
                    sample_ids.extend(row["id"] for row in rows)
                    scale = 1 / (args.accumulation * len(rows))
                    if choice_group_backward is not None:
                        loss_sum += choice_group_backward(
                            model, rows, args.candidate_batch, args.brier_weight, scale,
                            timings=timings, audit=audit_choice) * scale
                    else:
                        for row in rows:
                            loss_sum += choice_backward(
                                model, row, args.candidate_batch, args.brier_weight, scale,
                                timings=timings, audit=audit_choice) * scale
            if parallel is not None:
                with measure_phase(timings, "gradient_communication", cuda=True):
                    parallel.synchronize_gradients(parameters)
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            if audit_records is not None:
                audit_steps.append({"step": step + 1, "sample_ids": sample_ids, "learning_rates": learning_rates,
                                    "loss": loss_sum, "choices": audit_records, "gradients": record_gradients(trainable)})
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite gradients")
            with measure_phase(timings, "optimizer", cuda=True):
                optimizer.step()
            if parallel is None:
                peak_gpu_gib = torch.cuda.max_memory_allocated() / 2**30
                peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
            else:
                peak_gpu_gib, peak_reserved_gib = parallel.peak_memory()
                if timings is not None:
                    timings = parallel.max_phase_timings(timings)
            entry = {"step": step + 1, "loss": loss_sum, "gradient_norm": float(norm), "sample_ids": sample_ids,
                     "elapsed_seconds": time.time() - started, "peak_gpu_gib": peak_gpu_gib,
                     "peak_gpu_reserved_gib": peak_reserved_gib}
            if timings is not None:
                entry["phase_seconds"] = timings
            if prefix_usage is not None:
                entry["prefix_usage"] = prefix_usage.copy()
            if new_sft:
                scheduler.step()
                effective_decisions += step_decisions
                entry.update(learning_rates=dict(zip(("lora", "sid_embedding", "head"), learning_rates)),
                             unique_training_positions=len(seen_positions),
                             position_coverage=len(seen_positions) / data.train_positions,
                             effective_sid_decisions=step_decisions, cumulative_effective_sid_decisions=effective_decisions)
                if validation_users and ((step + 1) % args.validation_every == 0 or step + 1 == args.steps):
                    print(json.dumps({"event": "validation_started", "step": step + 1,
                                      "users": len(validation_users)}), flush=True)
                    ndcg = validation_ndcg(model, data, validation_users,
                                           args.validation_candidate_batch or args.candidate_batch,
                                           args.validation_backend)
                    entry["validation_ndcg_at10"] = ndcg
                    if best_validation is None or ndcg > best_validation["ndcg_at10"]:
                        best_validation = {"step": step + 1, "ndcg_at10": ndcg}
                        best_parameters = {name: p.detach().cpu().clone() for name, p in trainable.items()}
            if gspo:
                if new_rollout:
                    sampled_contexts += len(group_metrics)
                    sampled_context_ids.update(sample_ids)
                    hit_groups += sum(group["positive_code_rate"] > 0 for group in group_metrics)
                    nonzero_advantage_groups += sum(not group["zero_advantage_group"] for group in group_metrics)
                steps_with_policy_gradient += any(group["policy_logit_grad_norm"] > 0 for group in group_metrics)
                rollout_diagnostics = {"sampled_contexts": sampled_contexts,
                                       "unique_sampled_contexts": len(sampled_context_ids),
                                       "hit_groups": hit_groups, "zero_hit_groups": sampled_contexts - hit_groups,
                                       "hit_group_rate": hit_groups / sampled_contexts,
                                       "nonzero_advantage_groups": nonzero_advantage_groups,
                                       "nonzero_advantage_group_rate": nonzero_advantage_groups / sampled_contexts,
                                       "steps_with_policy_gradient": steps_with_policy_gradient}
                entry.update(objective="gspo", rollout_update=step % args.rollout_updates + 1, groups=group_metrics)
                entry.update(rollout_diagnostics)
            if hasattr(model, "sid_manifest"):
                sid_gradient = model.backbone.get_input_embeddings().sid_weight.grad
                entry["sid_embedding_gradient_norm"] = float(sid_gradient.norm()) if sid_gradient is not None else 0.0
            if args.full_validation and step + 1 == midpoint:
                with staged_output(candidate_path(args.output, midpoint)) as stage:
                    save_calibrated_checkpoint(model, data, args, stage, normalization, metadata["sid_encoding"])
                entry["validation_candidate"] = midpoint
            if log is not None:
                line = json.dumps(entry)
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)
    training_finished = time.time()
    if parallel is not None:
        parallel.close()
    if not primary:
        return
    if best_validation is not None:
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(best_parameters[name])
        del best_parameters, parameter
    if args.full_validation:
        checkpoint = candidate_path(args.output, args.steps)
        with staged_output(checkpoint) as stage:
            temperature, calibration_row, reference = save_calibrated_checkpoint(
                model, data, args, stage, normalization, metadata["sid_encoding"])
    else:
        checkpoint = args.output / "checkpoint"
        temperature, calibration_row, reference = save_calibrated_checkpoint(
            model, data, args, checkpoint, normalization, metadata["sid_encoding"])
    changed = {}
    if initial_parameters:
        for kind in ("lora", "head", "sid_embedding"):
            deltas = [(p.detach().cpu() - initial_parameters[name]).abs().max().item()
                      for name, p in model.named_parameters() if name in initial_parameters and
                      ((kind == "head" and name.startswith("head.")) or (kind == "lora" and ".lora_" in name)
                       or (kind == "sid_embedding" and name.endswith(".sid_weight")))]
            changed[kind + "_max_abs_change"] = max(deltas, default=0.0)
        del initial_parameters
    if new_sft:
        del scheduler
    del optimizer, parameters, trainable, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_model(checkpoint, args.base_model, item_catalog=args.data / "item_semantic_ids.jsonl")
    check = score(restored, calibration_row, args.candidate_batch).cpu()
    error = float((check - reference).abs().max())
    if not math.isfinite(error) or error > 0.05:
        raise ValueError(f"checkpoint reload mismatch: {error}")
    summary = {"status": "complete", "steps": args.steps, "training_decisions": args.steps * args.accumulation,
               "temperature": temperature, "reload_max_abs_error": error, "elapsed_seconds": time.time() - started,
               "startup_seconds": started - startup_started, "training_seconds": training_finished - started,
               "post_train_seconds": time.time() - training_finished,
               "note": "fixed-step adaptation; recommendation quality requires separate autoregressive evaluation"}
    summary.update(sid_encoding=metadata["sid_encoding"], history_memory=metadata["history_memory"], **changed)
    if prefix_usage is not None:
        summary["prefix_usage"] = prefix_usage
    if new_sft:
        summary.update(training_decisions=effective_decisions, effective_sid_decisions=effective_decisions,
                       unique_training_positions=len(seen_positions), position_coverage=len(seen_positions) / data.train_positions)
    if best_validation is not None:
        summary.update(best_validation=best_validation, unique_training_positions=len(seen_positions))
        summary["note"] = "checkpoint selected on validation; recommendation quality requires separate test evaluation"
        metadata["best_validation"] = best_validation
    if gspo:
        summary.pop("training_decisions")
        summary.update(objective="gspo", optimizer_updates=args.steps, context_updates=args.steps * args.accumulation,
                       sampled_trajectories=sampled_contexts * args.group_size, **rollout_diagnostics)
    if args.full_validation:
        metadata["validation_candidates"] = candidate_fingerprints(args.output, args.steps)
        summary["note"] = "SFT and GSPO candidates require full validation before publishing the test checkpoint"
    if audit_steps:
        torch.save(audit_steps, args.output / "sft_audit.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    metadata["status"] = "complete"
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
