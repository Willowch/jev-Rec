"""The native TIGER generator shared by MovieLens, Amazon and LETTER."""

import random
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TIGER_MODEL_DEFAULTS = {
    "num_layers": 4,
    "num_decoder_layers": 4,
    "d_model": 128,
    "d_ff": 1024,
    "num_heads": 6,
    "d_kv": 64,
    "dropout_rate": 0.1,
    "vocab_size": 1025,
    "pad_token_id": 0,
    "eos_token_id": 0,
    "feed_forward_proj": "relu",
}


class TIGER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = T5ForConditionalGeneration(
            T5Config(**config, decoder_start_token_id=config["pad_token_id"]))

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_loader, optimizer, device, *, max_steps=None):
    model.train()
    steps = len(train_loader)
    batches = train_loader
    if max_steps is not None:
        steps = min(max_steps, steps)
        batches = islice(train_loader, steps)
    total_loss = 0.0
    for batch in tqdm(batches, total=steps, desc="Training"):
        optimizer.zero_grad()
        loss, _ = model(input_ids=batch["history"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["target"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / steps
