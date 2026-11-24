"""Load TIGER text features and fit LETTER's train-only collaborative features."""

import math
from copy import copy

import numpy as np
import torch

from comparison.features import read_sentence_vectors, sentence_t5
from comparison.protocol import SEED, write_json
from comparison.recbole_runner import (SMOKE_BATCH_SIZE, build_training_dataset,
                                       make_config, shared_training, train_loader)


CF_DIM = 32


def encode_text(args, items, manifest, device):
    path, evidence = sentence_t5(args.data_dir, args.dataset, items, manifest, args.sentence_model,
                                 device, args.embedding_batch_size)
    return read_sentence_vectors(path, items), evidence


def train_cf(args, users, items, manifest, run_dir):
    from recbole.utils import get_model, init_seed

    directory = run_dir / "cf"
    directory.mkdir()
    args = copy(args)
    args.run_dir = run_dir
    setup = {"epochs": 1 if args.smoke else (args.cf_epochs or 100), "eval_step": 1, "patience": 1,
             "train_batch_size": SMOKE_BATCH_SIZE if args.smoke else 128}
    parameters = {"learning_rate": 0.001, "hidden_size": CF_DIM, "inner_size": 4 * CF_DIM,
                  "hidden_dropout_prob": 0.2, "attn_dropout_prob": 0.2}
    args.recbole_data, name, count = shared_training(
        args.data_dir, args.dataset, users, items, manifest, False, args.smoke)
    config = make_config("SASRec", name, args, setup, directory, parameters)
    init_seed(SEED, True)
    dataset = build_training_dataset(config)
    if len(dataset) != count:
        raise ValueError("RecBole changed the number of canonical training prefixes")
    loader = train_loader(config, dataset)
    model = get_model("SASRec")(config, dataset).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    model.train()
    history = []
    for epoch in range(1, setup["epochs"] + 1):
        total = torch.zeros((), device=config["device"])
        for interaction in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.calculate_loss(interaction.to(config["device"]))
            loss.backward()
            optimizer.step()
            total += loss.detach()
        average = total.item() / len(loader)
        if not math.isfinite(average):
            raise FloatingPointError(f"SASRec produced a non-finite loss at epoch {epoch}")
        history.append({"epoch": epoch, "loss": average, "updates": len(loader)})
        print(f"SASRec epoch {epoch}: {average:.6f} ({len(loader)} updates)", flush=True)

    tokens = dataset.field2token_id[model.ITEM_ID]
    if set(tokens) - {"[PAD]"} != set(items):
        raise ValueError("RecBole item IDs differ from the full comparison catalog")
    item_rows = np.array([tokens[item] for item in items])
    support = torch.zeros(len(tokens), dtype=torch.bool)
    support[dataset.inter_feat[model.ITEM_ID].flatten()] = True
    support[dataset.inter_feat[model.ITEM_SEQ].flatten()] = True
    warm = support.numpy()[item_rows]
    aligned = model.item_embedding.weight.detach().cpu().numpy()[item_rows]
    aligned[~warm] = 0.0
    if len(items) != len(aligned) or warm.sum() < 2 or not np.isfinite(aligned).all():
        raise ValueError("Expected aligned finite collaborative vectors with training support")
    np.savez(directory / "embeddings.npz", item_ids=np.asarray(items), embeddings=aligned, warm=warm)
    torch.save(model.state_dict(), directory / "sasrec.pth")
    write_json(directory / "training.json", {"train_pairs": count, "warm_items": int(warm.sum()),
                                             "batch_size": setup["train_batch_size"],
                                             "parameters": parameters, "epochs": history})
    return aligned, warm
