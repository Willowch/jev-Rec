"""Train RecBole's native models on OpenJev prefixes and score the shared cohort."""

import argparse
import logging
import math
import os
from contextlib import chdir
from itertools import islice, product
from pathlib import Path

from .prepare import load_prepared
from .protocol import (COMPARISON_DATASETS, MAX_HISTORY, PILOT_DATASET, ROOT, SEED, TOP_K, VALIDATION_INTERVAL, VALIDATION_K,
                       holdout, load_cohort, ranked_entry, read_json, require_full_validation,
                       require_training_selection, sampled_users, sha256, staged_output, TRAIN_SELECTION,
                       train_position, validation_metrics, validation_record,
                       validation_users, write_evaluation, write_json)


TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 64
SMOKE_BATCH_SIZE = 16
GRID = tuple(product((0.001, 0.0003), (0.2, 0.5)))


def training_rows(users, bert):
    for user in users:
        items = user["items"]
        position = train_position(user)
        # BERT4Rec's native transform masks the target appended to the history.
        sequence = items[max(0, position - MAX_HISTORY):position + int(bert)]
        yield user["user_id"], items[position], sequence


def export_training(directory, name, users, items, bert, smoke):
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / f"{name}.item").open("x", encoding="utf-8") as handle:
        handle.write("item_id:token\n")
        handle.writelines(item + "\n" for item in items)
    rows = training_rows(sampled_users(users, 128) if smoke else users, bert)
    if smoke:
        rows = islice(rows, 3 * SMOKE_BATCH_SIZE)
    count = 0
    with (directory / f"{name}.train.inter").open("x", encoding="utf-8") as handle:
        handle.write("user_id:token\titem_id:token\titem_id_list:token_seq\n")
        for user, target, sequence in rows:
            handle.write(f"{user}\t{target}\t{' '.join(sequence)}\n")
            count += 1
    return count


def make_config(model, name, args, setup, checkpoint_dir, parameters):
    from recbole.config import Config

    return Config(model=model, dataset=name, config_file_list=[str(ROOT / model / "model.yaml")],
                  config_dict={"data_path": str(getattr(args, "recbole_data", args.run_dir / "recbole")),
                               "checkpoint_dir": str(checkpoint_dir),
                               "benchmark_filename": ["train"],
                               "load_col": {"inter": ["user_id", "item_id", "item_id_list"],
                                            "item": ["item_id"]},
                               "alias_of_item_id": ["item_id_list"], "train_neg_sample_args": None,
                               "eval_args": {"order": "TO", "mode": "full"},
                               "valid_metric": "NDCG@10", "metrics": ["Recall", "NDCG"], "topk": [5, 10],
                               "train_batch_size": setup["train_batch_size"],
                               "eval_batch_size": args.eval_batch_size, "worker": 0,
                               "seed": SEED, "reproducibility": True, "learner": "adam",
                               "epochs": setup["epochs"], "eval_step": setup["eval_step"],
                               "stopping_step": setup["patience"], "log_wandb": False,
                               "use_gpu": bool(args.gpu_id), "gpu_id": args.gpu_id,
                               **parameters})


def train_loader(config, dataset):
    from recbole.data.dataloader import TrainDataLoader

    return TrainDataLoader(config, dataset, None, shuffle=True)


def recommendation_entries(model, dataset, users, items, split, batch_size, top_k):
    import torch
    from recbole.data.interaction import Interaction

    device = next(model.parameters()).device
    item_ids = dataset.field2token_id[model.ITEM_ID]
    if set(items) != set(item_ids) - {"[PAD]"}:
        raise ValueError("RecBole remapping changed the full catalog")
    catalog_columns = torch.tensor([item_ids[item] for item in items], device=device)
    columns = {item: index for index, item in enumerate(items)}
    bert = model.__class__.__name__ == "BERT4Rec"
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            sequences, lengths, targets, prior_rows, prior_cols = [], [], [], [], []
            for row, user in enumerate(batch):
                history, target = holdout(user, split)
                sequence = [item_ids[item] for item in history[-MAX_HISTORY:]]
                if bert:
                    # Native reconstruct_test_data discards the first slot after appending [MASK].
                    # Reserve that slot so it discards padding and keeps the entire input window.
                    sequence.insert(0, 0)
                lengths.append(len(sequence))
                sequences.append(sequence + [0] * (model.max_seq_length - len(sequence)))
                targets.append(target)
                seen = {columns[item] for item in history}
                prior_rows.extend([row] * len(seen))
                prior_cols.extend(seen)

            interaction = Interaction({model.ITEM_SEQ: torch.tensor(sequences, device=device),
                                       model.ITEM_SEQ_LEN: torch.tensor(lengths, device=device)})
            scores = model.full_sort_predict(interaction).index_select(1, catalog_columns)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("RecBole returned non-finite catalog scores")
            scores[prior_rows, prior_cols] = -torch.inf
            candidates = min(top_k, len(items))
            values, positions = torch.topk(scores, candidates, sorted=True)
            # Ties use the original dataset item ordering, even at the cutoff boundary.
            cutoff = values[:, -1:]
            boundary_ties = ((scores >= cutoff).sum(dim=1) > candidates) & cutoff[:, 0].isfinite()
            if boundary_ties.any():
                exact_positions = scores[boundary_ties].argsort(dim=1, descending=True, stable=True)[:, :candidates]
                positions[boundary_ties] = exact_positions
            positions = positions.sort(dim=1).values
            values = scores.gather(1, positions)
            order = values.argsort(dim=1, descending=True, stable=True)
            for user, target, indices, row_values in zip(
                    batch, targets, positions.gather(1, order).tolist(), values.gather(1, order).tolist()):
                recommendations = [{"item_id": items[index], "score": score}
                                   for index, score in zip(indices, row_values) if score != -torch.inf]
                yield ranked_entry(user["user_id"], target, recommendations)


def fit_candidate(model, config, dataset, validation_users, items, run_dir, args):
    import torch
    from recbole.trainer import Trainer
    from recbole.utils import get_model, init_seed

    init_seed(SEED, True)
    loader = train_loader(config, dataset)
    network = get_model(model)(config, dataset).to(config["device"])
    # RecBole creates its TensorBoard directory relative to the current directory.
    with chdir(run_dir):
        trainer = Trainer(config, network)
    best, history, misses = None, [], 0
    for epoch in range(1, config["epochs"] + 1):
        total_loss = trainer._train_epoch(loader, epoch - 1)
        loss = float(total_loss) / len(loader)
        if not math.isfinite(loss):
            raise FloatingPointError(f"{model} epoch {epoch}: non-finite loss")
        record = {"epoch": epoch, "average_batch_loss": loss, "optimizer_steps": len(loader)}
        if epoch % config["eval_step"] == 0 or epoch == config["epochs"]:
            entries = recommendation_entries(network, dataset, validation_users, items, "validation",
                                             args.eval_batch_size, VALIDATION_K)
            validation = validation_metrics(entries, validation_users)
            record.update(validation)
            if best is None or validation["NDCG@10"] > best["NDCG@10"]:
                best = {"epoch": epoch, **validation}
                torch.save(network.state_dict(), run_dir / "best.pth")
                misses = 0
            else:
                misses += 1
        history.append(record)
        print(f"{model} {run_dir.name}: {record}", flush=True)
        if misses >= config["stopping_step"]:
            break
    trainer.tensorboard.close()
    write_json(run_dir / "training.json", {"epochs": history, "best": best})
    return best


def training_files(directory, name):
    data = directory / name
    return {path.name: sha256(path) for path in (data / f"{name}.item", data / f"{name}.train.inter")}


def shared_training(data_dir, dataset, users, items, manifest, bert, smoke):
    schema = "masked" if bert else "next_item"
    name = "training"
    destination = data_dir.resolve() / "features" / "recbole" / TRAIN_SELECTION / schema / ("smoke" if smoke else "formal")
    if not destination.exists():
        try:
            with staged_output(destination) as stage:
                count = export_training(stage / name, name, users, items, bert, smoke)
                write_json(stage / "manifest.json", {"identity": manifest["identity"], "count": count,
                                                     "train_selection": TRAIN_SELECTION,
                                                     "files": training_files(stage, name)})
        except FileExistsError:
            if not destination.is_dir():
                raise
    recorded = read_json(destination / "manifest.json")
    expected = min(3 * SMOKE_BATCH_SIZE, len(users)) if smoke else manifest["train_pairs"]
    if (recorded.get("train_selection") != TRAIN_SELECTION
            or recorded["identity"] != manifest["identity"] or recorded["count"] != expected
            or recorded["files"] != training_files(destination, name)):
        raise ValueError("Shared RecBole training data differs from the prepared dataset")
    return destination, name, expected


def build_training_dataset(config):
    from recbole.data import create_dataset

    dataset, = create_dataset(config).build()
    return dataset


def choose_gpu(device):
    if device == "cpu":
        return ""
    index = int(device.partition(":")[2] or 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    devices = [entry.strip() for entry in visible.split(",")]
    if index >= len(devices) or devices[index] in ("", "-1"):
        raise ValueError(f"{device} is outside the GPUs visible to this process")
    return devices[index]


def run(model, args):
    args.gpu_id = choose_gpu(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    import recbole
    import torch
    from recbole.utils import get_model

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or a CUDA-enabled PyTorch installation.")
    torch.set_num_threads(4)
    root = args.run_dir.resolve()
    args.run_dir = root
    users, items, manifest = load_prepared(args.data_dir, args.dataset)
    selected, metadata = load_cohort(args.cohort, args.dataset, users, items, MAX_HISTORY)
    name = f"openjev_{args.dataset}_{model.lower()}"
    args.recbole_data = root / "recbole"
    config_hash = sha256(ROOT / model / "model.yaml")
    output = args.output or root / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)

    if args.evaluate_only:
        summary = read_json(root / "summary.json")
        selection = summary.get("validation_protocol")
        if metadata["scope"] == "formal":
            require_training_selection(summary)
            require_full_validation(selection, users)
        if summary.get("shared_training"):
            args.recbole_data, name, _ = shared_training(
                args.data_dir, args.dataset, users, items, manifest, model == "BERT4Rec",
                metadata["scope"] == "smoke")
        if (summary["identity"] != manifest["identity"] or summary["model"] != model
                or summary["model_config_sha256"] != config_hash or summary["scope"] != metadata["scope"]
                or summary["training_files"] != training_files(args.recbole_data, name)):
            raise ValueError("checkpoint, trial configuration, data or cohort scope differ from training")
        setup = summary["setup"]
        winner = summary["winner"]
        config = make_config(model, name, args, setup, root / winner["trial"], winner["parameters"])
        data = build_training_dataset(config)
    else:
        if args.smoke != (metadata["scope"] == "smoke"):
            raise ValueError("match --smoke to the scope of the evaluation cohort")
        root.mkdir(parents=True, exist_ok=False)
        setup = {"epochs": 100, "eval_step": args.eval_step, "patience": 5,
                 "train_batch_size": args.train_batch_size}
        grid = GRID
        if args.smoke:
            setup.update(epochs=1, eval_step=1, train_batch_size=SMOKE_BATCH_SIZE)
            grid = GRID[:1]
        elif metadata["scope"] == "pilot":
            setup.update(epochs=args.epochs or 20)
            grid = GRID[:args.trials or 1]
        args.recbole_data, name, count = shared_training(
            args.data_dir, args.dataset, users, items, manifest, model == "BERT4Rec", args.smoke)
        selected_validation = validation_users(users, smoke=args.smoke, pilot=metadata["scope"] == "pilot")
        selection = validation_record(selected_validation, smoke=args.smoke)
        trials, data = [], None
        for index, (learning_rate, dropout) in enumerate(grid, 1):
            trial_dir = root / f"trial-{index}"
            trial_dir.mkdir()
            parameters = {"learning_rate": learning_rate, "hidden_dropout_prob": dropout,
                          "attn_dropout_prob": dropout}
            config = make_config(model, name, args, setup, trial_dir, parameters)
            keys = ("seed", "epochs", "eval_step", "stopping_step", "train_batch_size", "eval_batch_size",
                    "learner", "learning_rate", "loss_type", "n_layers", "n_heads", "hidden_size", "inner_size",
                    "hidden_dropout_prob", "attn_dropout_prob", "MAX_ITEM_LIST_LENGTH", "transform",
                    "mask_ratio", "ft_ratio", "device")
            effective = {key: config[key] for key in keys if key in config}
            effective["device"] = str(effective["device"])
            write_json(trial_dir / "config.json", effective)
            if data is None:
                data = build_training_dataset(config)
                if len(data) != count:
                    raise ValueError("RecBole changed the number of training examples")
            best = fit_candidate(model, config, data, selected_validation, items, trial_dir, args)
            trials.append({"trial": trial_dir.name, "parameters": parameters, "validation": best})
        winner = max(trials, key=lambda trial: trial["validation"]["NDCG@10"])
        config = make_config(model, name, args, setup, root / winner["trial"], winner["parameters"])
        write_json(root / "summary.json", {"model": model, "scope": metadata["scope"], "setup": setup,
                                           "identity": manifest["identity"], "train_pairs": count,
                                           "train_selection": TRAIN_SELECTION,
                                           "model_config_sha256": config_hash,
                                           "shared_training": True,
                                           "training_files": training_files(args.recbole_data, name),
                                           "recbole": recbole.__version__, "torch": torch.__version__,
                                            "trials": trials, "winner": winner, "seed": SEED,
                                            "validation_protocol": selection})

    if model == "BERT4Rec":
        # The native model reads field names registered by the training transform at construction.
        from recbole.data.transform import construct_transform

        construct_transform(config)
    network = get_model(model)(config, data).to(config["device"])
    state = torch.load(root / winner["trial"] / "best.pth", map_location=config["device"], weights_only=True)
    network.load_state_dict(state)
    entries = recommendation_entries(network, data, selected, items, "test", args.eval_batch_size, TOP_K)
    result = write_evaluation(output, entries, selected, items, metadata, model,
                              protocol="full catalog, seen items excluded, exact scoring", seed=SEED,
                              validation_protocol=selection, train_selection=summary.get("train_selection")
                              if args.evaluate_only else TRAIN_SELECTION)
    print(f"{model} {args.dataset}: saved {result['users']} user rankings to {output}", flush=True)


def main(model):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=COMPARISON_DATASETS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--eval-step", type=int, default=VALIDATION_INTERVAL, help="Epochs between validations")
    parser.add_argument("--epochs", type=int, help="Training epochs for the MovieLens-100K pilot")
    parser.add_argument("--trials", type=int, choices=range(1, len(GRID) + 1), help="Trial count for the pilot")
    args = parser.parse_args()
    if args.device != "cpu" and not (args.device == "cuda" or args.device.startswith("cuda:")
                                     and args.device[5:].isdecimal()):
        parser.error("device must be cpu, cuda or cuda:N")
    if min(args.train_batch_size, args.eval_batch_size, args.eval_step) < 1:
        parser.error("batch sizes and evaluation interval must be positive")
    if args.evaluate_only and args.output is None:
        parser.error("--evaluate-only requires a new --output path")
    if (args.epochs is not None or args.trials is not None) and args.dataset != PILOT_DATASET:
        parser.error("--epochs and --trials are for the MovieLens-100K pilot")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    logging.basicConfig(level=logging.INFO)
    run(model, args)
