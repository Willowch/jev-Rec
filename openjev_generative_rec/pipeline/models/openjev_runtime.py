"""复用官方 DecisionModel；分块计算完整的 255 候选损失。"""
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ..common import ROOT, sha256
from ..data_preparation.build_jev_data import read_ids, sid_layout

sys.path.insert(0, str(ROOT / "third_party/Open-Jev"))


def load_model(checkpoint, base_model=None, train=False, sid_encoding="auto",
               history_memory="auto", item_catalog=None, *, device="cuda:0", gradient_checkpointing=True):
    from jev.model import DecisionModel
    from peft import PeftModel
    from .sid_tokens import FORMAT, IDENTITY_FORMAT, LayeredSIDMixin, install_sid_embeddings
    from .history_memory import load_history_memory
    # 用同一参考路径复现未安装 FLA 时的数值；默认自动使用已安装的加速内核。
    if os.environ.get("OPENJEV_REFERENCE_KERNELS") == "1":
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        qwen.chunk_gated_delta_rule = None
        qwen.fused_recurrent_gated_delta_rule = None
        qwen.FusedRMSNormGated = None
        qwen.causal_conv1d_fn = None
        qwen.causal_conv1d_update = None
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "model.json").read_text())
    saved_encoding = config.get("sid_encoding", "text")
    if saved_encoding not in ("text", FORMAT, IDENTITY_FORMAT) or sid_encoding not in ("auto", "text", "layered"):
        raise ValueError("unsupported SID encoding")
    has_sid = (checkpoint / "sid_tokens.json").exists()
    if has_sid != (saved_encoding != "text"):
        raise ValueError("SID checkpoint metadata is missing or inconsistent")
    if has_sid and sid_encoding == "text":
        raise ValueError("cannot load a layered SID checkpoint as legacy text")
    layered = has_sid or sid_encoding == "layered" or (train and sid_encoding == "auto")
    identities = sid_layout(read_ids(item_catalog))[1] if item_catalog is not None else []
    if (saved_encoding == IDENTITY_FORMAT and not identities) or (
        identities and (not layered or saved_encoding == FORMAT or (not train and not has_sid))
    ):
        raise ValueError("Identity SIDs require a matching four-layer checkpoint or fresh layered SFT")
    fingerprint = sha256(item_catalog) if identities else None
    if saved_encoding == IDENTITY_FORMAT:
        manifest = json.loads((checkpoint / "sid_tokens.json").read_text())
        if manifest.get("catalog_sha256") != fingerprint:
            raise ValueError("Identity SID checkpoint belongs to a different item mapping")
    memory = load_history_memory(checkpoint, config, history_memory, item_catalog)
    if memory is not None and not layered:
        raise ValueError("history memory requires layered SID encoding")
    if train and layered and not has_sid:
        print("Initializing layered SID embeddings from legacy checkpoint; run SFT before GSPO.", flush=True)
    model_path = Path(base_model or config["model_id"])
    if model_path.is_dir() and (model_path / "download_manifest.json").exists():
        downloaded = json.loads((model_path / "download_manifest.json").read_text())
        if downloaded["revision"] != config["revision"]:
            raise ValueError("基础模型版本与 OpenJev checkpoint 不一致")
    class LayeredDecisionModel(LayeredSIDMixin, DecisionModel):
        pass
    model_class = LayeredDecisionModel if layered else DecisionModel
    model = model_class(str(Path(base_model).resolve()) if base_model else config["model_id"],
                        config["revision"], device=str(device), lora_rank=0, max_length=config["max_length"])
    if os.environ.get("OPENJEV_COMPACT_DELTA_RULE") == "1":
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        from .compact_delta_rule import compact_gated_delta_rule

        for layer in model.backbone.modules():
            if getattr(layer, "chunk_gated_delta_rule", None) is torch_chunk_gated_delta_rule:
                layer.chunk_gated_delta_rule = compact_gated_delta_rule
    model.backbone = PeftModel.from_pretrained(model.backbone, checkpoint / "adapter", is_trainable=train)
    model.lora_rank = config["lora_rank"]
    model.head.load_state_dict(torch.load(checkpoint / "head.pt", map_location=str(device), weights_only=True))
    model.head.requires_grad_(train)
    if layered:
        install_sid_embeddings(model, checkpoint if has_sid else None, train=train,
                               identities=identities, catalog_sha256=fingerprint)
        model.history_memory = memory
        if os.environ.get("OPENJEV_CPU_TOKEN_EMBEDDINGS") == "1":
            embedding = model.backbone.get_input_embeddings()
            embedding.base.cpu()
            torch.cuda.empty_cache()
    if train:
        if gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if gradient_checkpointing == "partial":
                # Keep three of every four blocks checkpointed to bound activation memory.
                for layer in model.backbone.get_base_model().layers[::4]:
                    layer.gradient_checkpointing = False
        else:
            model.backbone.gradient_checkpointing_disable()
        model.backbone.enable_input_require_grads()
    set_training_mode(model, train)
    return model


def set_training_mode(model, train):
    # Gradient checkpointing needs train mode; the two scoring passes need matching dropout.
    model.train(train)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def chunks(row, size):
    for start in range(0, len(row["options"]), size):
        yield start, {**row, "options": row["options"][start:start + size]}


@contextmanager
def measure_phase(timings, phase, *, cuda=False):
    if timings is None:
        yield
        return
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    yield
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    timings[phase] = timings.get(phase, 0.0) + time.perf_counter() - started


def prepare_candidate_chunks(model, row, chunk_size):
    prepare_record = getattr(model, "prepare_record", None)
    if prepare_record is not None:
        row = prepare_record(row)
    return [(start, model.encode_candidates([part])) for start, part in chunks(row, chunk_size)]


@torch.no_grad()
def score(model, row, chunk_size=1, *, prepared=None):
    if prepared is not None:
        return torch.cat([model.score_encoded(*inputs)[0].float() for _, inputs in prepared])
    return torch.cat([model([part])[0].float() for _, part in chunks(row, chunk_size)])


def backward_logits(model, row, gradient, chunk_size=1, *, prepared=None, timings=None):
    """将外部损失的 dL/dlogits 分块传回 LoRA；共享前缀可先合并梯度。"""
    use_encoded = prepared is not None
    parts = prepared if use_encoded else chunks(row, chunk_size)
    for start, part in parts:
        count = part[1][0] if use_encoded else len(part["options"])
        grad = gradient[start:start + count]
        if torch.count_nonzero(grad) == 0:
            continue
        with measure_phase(timings, "gradient_forward", cuda=True):
            values = (model.score_encoded(*part) if use_encoded else model([part]))[0].float()
        with measure_phase(timings, "backward", cuda=True):
            values.backward(grad)


def choice_loss(logits, row, brier_weight):
    target = torch.tensor(row["target"], device=logits.device)
    loss = -(target * logits.log_softmax(-1)).sum()
    loss = loss + brier_weight * (logits.softmax(-1) - target).square().sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    return loss


def backward_choice(model, row, chunk_size=1, brier_weight=0.1, scale=1.0, *, reuse_encoding=False,
                    timings=None, audit=None, prepared=None):
    if row["target"] == [1.0]:
        return 0.0
    if prepared is None and reuse_encoding:
        with measure_phase(timings, "candidate_encoding"):
            prepared = prepare_candidate_chunks(model, row, chunk_size)
    if prepared is not None and len(prepared) == 1:
        _, inputs = prepared[0]
        with measure_phase(timings, "gradient_forward", cuda=True):
            logits = model.score_encoded(*inputs)[0].float()
        loss = choice_loss(logits, row, brier_weight)
        with measure_phase(timings, "backward", cuda=True):
            (loss * scale).backward()
    else:
        # Recompute one candidate batch at a time to bound activation memory.
        with measure_phase(timings, "scoring_forward", cuda=True):
            logits = score(model, row, chunk_size, prepared=prepared).detach().requires_grad_(True)
        loss = choice_loss(logits, row, brier_weight)
        gradient, = torch.autograd.grad(loss * scale, logits)
        backward_logits(model, row, gradient, chunk_size, prepared=prepared, timings=timings)
    if audit is not None:
        audit(row, logits)
    return float(loss.detach())
