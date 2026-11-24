"""使用 MiniLM 或 OpenJev 同版本 Qwen 文本骨干生成物品 embedding。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..common import enable_comparison


def load_items(zip_path: Path):
    enable_comparison()
    from comparison.sources import ml1m_movies

    for row in ml1m_movies(zip_path):
        yield row["item_id"], row["title"], row["genres"]


def mean_pool(hidden, mask):
    # Qwen 使用 bf16 推理，池化转为 float32，避免累加精度损失。
    hidden = hidden.float()
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


def load_encoder(path, device):
    config = AutoConfig.from_pretrained(path)
    if config.model_type == "qwen3_5":
        from transformers import AutoModelForImageTextToText
        full = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map={"": device}, attn_implementation="sdpa")
        total = sum(p.numel() for p in full.parameters())
        # 与 OpenJev 提取相同的文本骨干；不加载推荐 LoRA 或 decision head。
        model = full.model.language_model
        del full
        model.config.use_cache = False
    else:
        model = AutoModel.from_pretrained(path).to(device)
        total = sum(p.numel() for p in model.parameters())
    model.requires_grad_(False).eval()
    return model, total, config.model_type


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ml1m", type=Path)
    source.add_argument("--comparison-data", type=Path)
    parser.add_argument("--dataset", choices=("ml1m",), default="ml1m",
                        help="Dataset for --comparison-data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="models/qwen3.5-2b")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.comparison_data:
        enable_comparison()
        from comparison.prepare import load_metadata, load_prepared

        _, ids, manifest = load_prepared(args.comparison_data, args.dataset)
        rows = load_metadata(args.comparison_data, ids, manifest)
    else:
        rows = [{"item_id": item, "title": title, "genres": genres, "text": f"{title}. Genres: {genres}."}
                for item, title, genres in load_items(args.ml1m)]
    texts = [row["text"] for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, total_parameters, model_type = load_encoder(args.model, device)
    parameters = sum(p.numel() for p in model.parameters())
    vectors = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="embedding"):
            batch = tokenizer(texts[start:start + args.batch_size], padding=True, truncation=True,
                              max_length=args.max_length, return_tensors="pt").to(device)
            extra = {"use_cache": False} if model_type == "qwen3_5" else {}
            vectors.append(mean_pool(model(**batch, **extra).last_hidden_state, batch["attention_mask"]).cpu().numpy())
    embeddings = np.concatenate(vectors).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding 含非有限值")
    np.save(args.out / "embeddings.npy", embeddings)
    with (args.out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"items": len(rows), "dim": int(embeddings.shape[1]), "device": device,
               "model": args.model, "parameters": parameters, "text": "title + genres",
               "pooling": "attention-masked mean + L2 normalization",
               "model_type": model_type, "full_model_parameters": total_parameters,
               "batch_size": args.batch_size, "max_length": args.max_length,
               "dtype": str(next(model.parameters()).dtype), "trainable_parameters": 0,
               "chat_template": False}
    provenance = Path(args.model) / "download_manifest.json"
    if provenance.exists():
        summary["model_provenance"] = json.loads(provenance.read_text())
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
