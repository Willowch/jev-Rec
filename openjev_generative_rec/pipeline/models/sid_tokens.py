"""在模型边界把原始 RQ 编号转为分层 token；单独训练新增 embedding 行。"""
import json
from pathlib import Path

import torch
from torch import nn

from ..common import sha256
from ..data_preparation.build_jev_data import IDENTITY_TOKEN, SEMANTIC_LAYERS, TOKENS

SID_TOKENS = [f"<sid{level}_{code:03d}>" for level in range(1, 4) for code in range(255)]
ITEM_START, ITEM_END = "<|sid_item|>", "<|sid_end|>"
ADDED_TOKENS = SID_TOKENS + [ITEM_START, ITEM_END]
FORMAT = "layered_sid_v1"
IDENTITY_FORMAT = "layered_sid_identity_v2"


def sid_token(token, level):
    if level == SEMANTIC_LAYERS + 1 and IDENTITY_TOKEN.fullmatch(token):
        return token
    if token not in TOKENS or level not in (1, 2, 3):
        raise ValueError(f"invalid SID or layer: {token}, {level}")
    return SID_TOKENS[(level - 1) * 255 + TOKENS.index(token)]


def encode_record(row, depth=SEMANTIC_LAYERS):
    """保持候选顺序及标签不变，只改模型可见的历史、前缀和候选。"""
    state = row["state"]
    level = state["semantic_id_layer"]
    if row["kind"] != "choice" or not 1 <= level <= depth:
        raise ValueError("Choice record layer differs from the SID checkpoint")
    history = []
    for item in state["history"].split():
        code = item.split("/")
        if len(code) != depth:
            raise ValueError("history item SID depth differs from the checkpoint")
        history.append(ITEM_START + "".join(sid_token(t, i + 1) for i, t in enumerate(code)) + ITEM_END)
    prefix = [] if state["target_prefix"] == "<empty>" else state["target_prefix"].split("/")
    if len(prefix) != level - 1:
        raise ValueError("target prefix length does not match semantic_id_layer")
    return {**row, "state": {**state, "history": " ".join(history),
                            "target_prefix": ITEM_START + "".join(sid_token(t, i + 1) for i, t in enumerate(prefix))},
            "options": [sid_token(token, level) for token in row["options"]]}


class SIDEmbedding(nn.Module):
    """独立 FP32 小表：AdamW 不持有原词表的梯度/优化器状态，也不会衰减原词表。"""
    def __init__(self, base, token_ids, initial, train):
        super().__init__()
        if len(set(token_ids)) != len(token_ids) or initial.shape != (len(token_ids), base.weight.shape[1]):
            raise ValueError("SID embedding shape or token IDs are invalid")
        self.base = base.requires_grad_(False)
        self.sid_weight = nn.Parameter(initial.to(device=base.weight.device, dtype=torch.float32).clone(),
                                       requires_grad=train)
        lookup = torch.full((max(base.num_embeddings, max(token_ids) + 1),), -1,
                            dtype=torch.long, device=base.weight.device)
        lookup[token_ids] = torch.arange(len(token_ids), device=lookup.device)
        self.register_buffer("sid_lookup", lookup, persistent=False)
        self.num_embeddings, self.embedding_dim = len(lookup), base.embedding_dim

    @property
    def weight(self):
        # 兼容 HF 对 embedding dtype/device 的读取；原表仍由基础模型文件提供。
        return self.base.weight

    def forward(self, input_ids):
        rows = self.sid_lookup[input_ids]
        mask = rows >= 0
        base_ids = input_ids.masked_fill(mask, 0)
        base_values = self.base(base_ids.to(self.base.weight.device)).to(input_ids.device)
        sid_values = torch.nn.functional.embedding(rows.clamp_min(0), self.sid_weight).to(base_values.dtype)
        return torch.where(mask.unsqueeze(-1), sid_values, base_values)


def validate_tokenizer(tokenizer, expected_ids, tokens=ADDED_TOKENS):
    actual = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]
    if actual != [[i] for i in expected_ids] or len(set(expected_ids)) != len(tokens):
        raise ValueError("SID tokens must each encode to one distinct, checkpoint-consistent token ID")


def install_sid_embeddings(model, checkpoint=None, train=False, *, identities=(), catalog_sha256=None):
    """在 PEFT adapter 加载后安装；新旧 checkpoint 均不扩展原基础权重文件。"""
    from transformers import AutoTokenizer
    base = model.backbone.get_input_embeddings()
    if base.weight.device.type == "meta":
        raise ValueError("layered SID embeddings require a resident input embedding (no meta offload)")
    checkpoint = Path(checkpoint) if checkpoint else None
    added_tokens = ADDED_TOKENS + list(identities)
    sid_format = IDENTITY_FORMAT if identities else FORMAT
    manifest_path = checkpoint / "sid_tokens.json" if checkpoint else None
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["format"] != sid_format or manifest["tokens"] != added_tokens:
            raise ValueError("unsupported SID checkpoint format or token order")
        if identities and manifest.get("catalog_sha256") != catalog_sha256:
            raise ValueError("Identity SID checkpoint belongs to a different item mapping")
        if manifest["base_vocab_size"] != base.num_embeddings:
            raise ValueError("SID checkpoint base vocabulary size mismatch")
        if sha256(checkpoint / "sid_embeddings.pt") != manifest["embedding_sha256"]:
            raise ValueError("SID embedding checkpoint checksum mismatch")
        if sha256(checkpoint / "tokenizer/tokenizer.json") != manifest["tokenizer_sha256"]:
            raise ValueError("SID tokenizer checksum mismatch")
        model.tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
        token_ids = manifest["token_ids"]
        initial = torch.load(checkpoint / "sid_embeddings.pt", map_location="cpu", weights_only=True)
    else:
        # 用旧文本片段均值初始化，层号参与初始化；不依赖任意随机向量。
        vocabulary = model.tokenizer.get_vocab()
        if any(token in vocabulary for token in added_tokens):
            raise ValueError("base tokenizer already contains SID tokens without a SID checkpoint")
        texts = [f"layer {level} sid_{code:03d}" for level in range(1, 4) for code in range(255)]
        texts += ["item start", "item end"]
        texts += [f"item identity rank {token[3:-1]}" for token in identities]
        with torch.no_grad():
            initial = torch.stack([base(torch.tensor(model.tokenizer.encode(text, add_special_tokens=False),
                                                     device=base.weight.device)).float().mean(0) for text in texts])
        # add_tokens(special_tokens=True) 跨 HF 版本可用，保持已有 chat 特殊 token 配置。
        model.tokenizer.add_tokens(added_tokens, special_tokens=True)
        token_ids = model.tokenizer.convert_tokens_to_ids(added_tokens)
        manifest = {"format": sid_format, "tokens": added_tokens, "token_ids": token_ids,
                    "base_vocab_size": base.num_embeddings, "initialization": "mean legacy text embedding with level"}
        if identities:
            manifest["catalog_sha256"] = catalog_sha256
    validate_tokenizer(model.tokenizer, token_ids, added_tokens)
    model.tokenizer.padding_side = "right"
    model.backbone.set_input_embeddings(SIDEmbedding(base, token_ids, initial, train))
    model.sid_manifest = manifest
    model.sid_depth = SEMANTIC_LAYERS + bool(identities)


class LayeredSIDMixin:
    """保留官方 DecisionModel 的候选 prompt、骨干和标量评分头。"""
    def prepare_record(self, row):
        memory = getattr(self, "history_memory", None)
        return encode_record(memory.enrich(row) if memory is not None else row, self.sid_depth)

    def forward(self, records):
        return super().forward([self.prepare_record(row) for row in records])

    def score_cached(self, records, *, batch_size=32):
        from jev.prefix_cache import score_cached

        if str(self.device_name) == "meta":
            raise ValueError("Layered prefix caching requires resident SID embeddings")
        return score_cached(self, [self.prepare_record(row) for row in records], batch_size=batch_size)

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output / "adapter", save_embedding_layers=False)
        torch.save(self.head.state_dict(), output / "head.pt")
        embedding = self.backbone.get_input_embeddings()
        torch.save(embedding.sid_weight.detach().cpu(), output / "sid_embeddings.pt")
        self.tokenizer.save_pretrained(output / "tokenizer")
        manifest = {**self.sid_manifest, "embedding_sha256": sha256(output / "sid_embeddings.pt"),
                    "tokenizer_sha256": sha256(output / "tokenizer/tokenizer.json")}
        (output / "sid_tokens.json").write_text(json.dumps(manifest, indent=2))
        config = {"model_id": self.model_id, "revision": self.revision,
            "max_length": self.max_length, "lora_rank": self.lora_rank, "sid_encoding": self.sid_manifest["format"],
            "method": "independent_candidate_lora_sid_embedding_nll_brier"}
        memory = getattr(self, "history_memory", None)
        if memory is not None:
            config["history_memory"] = memory.save(output)
        (output / "model.json").write_text(json.dumps(config, indent=2))
