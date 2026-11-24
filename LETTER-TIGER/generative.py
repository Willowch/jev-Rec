"""Train the TIGER backbone with LETTER's ranking-guided loss."""

import math

import torch
from torch.nn import functional as F
from torch.utils.data import Subset
from transformers import get_cosine_schedule_with_warmup

from amazon_eval import calibrate, recommendation_entries
from comparison.protocol import (MAX_HISTORY, PILOT_DATASET, SEED, TOP_K, VALIDATION_K, sampled_users, validation_metrics,
                                 validation_record, validation_users, write_evaluation, write_json)
from dataloader import GenRecDataLoader
from tiger_model import TIGER, TIGER_MODEL_DEFAULTS, set_seed
from ml1m_data import TrainingPairs, TrainingSequences
from ml1m_eval import PROTOCOL


TRAIN_TEMPERATURE = 0.7
LEARNING_RATES = (0.0005, 0.001)
PATIENCE_EPOCHS = 30


def ranking_loss(model, batch, device):
    labels = batch["target"].to(device)
    decoder_ids = F.pad(labels[:, :-1], (1, 0), value=model.config.decoder_start_token_id)
    logits = model(input_ids=batch["history"].to(device),
                   attention_mask=batch["attention_mask"].to(device),
                   decoder_input_ids=decoder_ids, use_cache=False).logits
    scores = logits.flatten(0, 1).float() / TRAIN_TEMPERATURE
    return F.cross_entropy(scores, labels.flatten())


@torch.inference_mode()
def validate(model, catalog, users, temperature, args, device):
    model.eval()
    entries = recommendation_entries(model, catalog, users, "validation", temperature, device,
                                     max_history=MAX_HISTORY, batch_size=args.eval_batch_size,
                                     beam_size=VALIDATION_K, top_k=VALIDATION_K)
    return validation_metrics(entries, users)


def fit_trial(args, data, catalog, validation_users, directory, learning_rate, device):
    directory.mkdir()
    set_seed(SEED)
    epochs = 1 if args.smoke else (args.generator_epochs or 100)
    batch_size = 16 if args.smoke else args.train_batch_size
    train_data = TrainingSequences(sampled_users(data.users, 128), data.manifest) if args.smoke else data
    pairs = TrainingPairs(train_data, catalog.tokens)
    if args.smoke:
        pairs = Subset(pairs, range(min(len(pairs), 3 * batch_size)))
    loader = GenRecDataLoader(pairs, batch_size=batch_size, num_workers=0)
    model = TIGER(TIGER_MODEL_DEFAULTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = len(loader) * epochs
    schedule = get_cosine_schedule_with_warmup(optimizer, int(steps * 0.01), steps)

    best, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = torch.zeros((), device=device)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(model.model, batch, device)
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.detach()
        average = total.item() / len(loader)
        if not math.isfinite(average):
            raise FloatingPointError(f"LETTER-TIGER produced a non-finite loss at epoch {epoch}")
        record = {"epoch": epoch, "loss": average, "updates": len(loader)}
        history.append(record)
        if epoch % args.eval_step == 0 or epoch == epochs:
            model.eval()
            calibration = calibrate(model, data, catalog, args, device)
            metrics = validate(model, catalog, validation_users, calibration["temperature"], args, device)
            record.update(metrics)
            if best is None or record["NDCG@10"] > best["validation"]["NDCG@10"]:
                best = {"trial": directory.name, "learning_rate": learning_rate,
                        "validation": {"epoch": epoch, **metrics}, "calibration": calibration}
                torch.save(model.state_dict(), directory / "best.pth")
        print(f"LETTER-TIGER lr={learning_rate:g} epoch {epoch}: {record}", flush=True)
        if best is not None and epoch - best["validation"]["epoch"] >= PATIENCE_EPOCHS:
            break

    write_json(directory / "training.json", {
        "train_pairs": len(pairs), "batch_size": batch_size, "checkpoint": best,
        "epochs": history, "ranking_temperature": TRAIN_TEMPERATURE,
    })
    return best


def train_generator(args, users, catalog, directory, device):
    data = TrainingSequences(users, {"max_history": MAX_HISTORY})
    selected = validation_users(users, smoke=args.smoke, pilot=args.dataset == PILOT_DATASET)
    args.seed = SEED
    args.calibration_rows = 3 if args.smoke else 30
    count = 1 if args.smoke else (args.trials or len(LEARNING_RATES))
    trials = [fit_trial(args, data, catalog, selected, directory / f"trial-{index}", lr, device)
              for index, lr in enumerate(LEARNING_RATES[:count], 1)]
    winner = max(trials, key=lambda trial: (trial["validation"]["NDCG@10"], -trial["validation"]["epoch"]))
    return trials, winner, validation_record(selected, smoke=args.smoke)


def test_checkpoint(args, directory, catalog, selected, metadata, summary, device, output):
    winner = summary["winner"]
    model = TIGER(summary["model_config"]).to(device)
    weights = directory / winner["trial"] / "best.pth"
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    temperature = winner["calibration"]["temperature"]
    entries = recommendation_entries(model, catalog, selected, "test", temperature, device,
                                     max_history=MAX_HISTORY, batch_size=args.eval_batch_size,
                                     beam_size=TOP_K, top_k=TOP_K)
    return write_evaluation(output, entries, selected, catalog.codes, metadata, "LETTER-TIGER",
                            seed=SEED, protocol=PROTOCOL, policy_normalization="full", beam_size=TOP_K,
                            temperature=temperature, validation_protocol=summary.get("validation_protocol"),
                            train_selection=summary.get("train_selection"))
