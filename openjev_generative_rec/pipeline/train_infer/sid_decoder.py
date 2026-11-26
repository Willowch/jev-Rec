"""Benchmark, train and evaluate parallel SID projections on the shared protocol."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from filelock import FileLock

from ..common import enable_comparison, sha256
from ..data_preparation.build_jev_data import SequenceData
from ..data_preparation.embedding_cache import encoder_fingerprint
from ..models.openjev_runtime import load_model, set_training_mode
from ..models.sid_decoder import SIDDecoder
from .gspo_selection import checkpoint_digest, require_sft_source
from .recommend import checkpoint_config
from .test_shards import merge_shards, read_shard, shard_specs

enable_comparison()
from comparison.protocol import (COUNTS, MAX_HISTORY, SEED, TOP_K, TRAIN_SELECTION, VALIDATION_K,
                                 checked_ranks, holdout, item_key, load_cohort, ranked_entry,
                                 read_evaluation, read_json, sampled_users, staged_output,
                                 train_position, validation_metrics, validation_record, write_evaluation, write_json)


METHOD = "openjev_sid_projection_v1"
FORMAL_DATASETS = ("ml1m", "Industrial_and_Scientific")
TEST_SHARD_SIZE = 256
PUBLISHED_EPOCHS = 10
PUBLISHED_MIN_EPOCHS = 5
PUBLISHED_PATIENCE = 3


def log(event, **data):
    print(json.dumps({"event": event, **data}, allow_nan=False), flush=True)


def require_formal_cohort(dataset, data, test_users, cohort):
    counts = (len(data.users), len(data.item_ids), len(test_users))
    if (dataset not in FORMAL_DATASETS or cohort["scope"] != "formal" or counts != COUNTS[dataset]):
        raise ValueError(f"Formal {dataset} needs the complete training, validation and test cohort and catalog")


def load_decision(args, checkpoint, *, train):
    initialize = args.initialization == "published" and checkpoint == args.source_checkpoint
    return load_model(checkpoint, args.base_model, train=train or initialize,
                      sid_encoding="layered" if initialize else "auto",
                      history_memory="on" if initialize else "auto",
                      item_catalog=args.sid_data / "item_semantic_ids.jsonl", device=args.device,
                      gradient_checkpointing=train and {"on": True, "partial": "partial", "off": False}[
                          args.gradient_checkpointing])


def new_decoder(args, data, checkpoint, *, train):
    torch.manual_seed(SEED)
    decision = load_decision(args, checkpoint, train=train)
    decoder = SIDDecoder(decision, data.catalog, MAX_HISTORY)
    set_training_mode(decoder, train)
    return decoder


def runtime_identity(args):
    return {"decode_backend": args.decode_backend, "decode_batch": args.decode_batch, "device": args.device,
            "runtime": {name: version(name) for name in ("torch", "transformers", "peft")},
            "encoder_options": {name: os.environ[name] for name in (
                "JEV_LOAD_4BIT", "JEV_LOAD_8BIT", "JEV_TORCH_DTYPE", "OPENJEV_REFERENCE_KERNELS",
                "OPENJEV_COMPACT_DELTA_RULE", "OPENJEV_CPU_TOKEN_EMBEDDINGS", "JEV_DEVICE_MAP") if name in os.environ}}


def optimizer_for(decision, heads=()):
    sid_weight = decision.backbone.get_input_embeddings().sid_weight
    backbone = [param for param in decision.backbone.parameters()
                if param.requires_grad and param is not sid_weight]
    return torch.optim.AdamW([{"params": backbone, "lr": 2e-5},
                             {"params": [sid_weight], "lr": 1e-4},
                             {"params": heads, "lr": 1e-4}], weight_decay=0.01)


def update(optimizer, parameters):
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory(device):
    if device.type != "cuda":
        return {}
    return {"peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def prepare_train(decoder, users):
    positions = [train_position(user) for user in users]
    prompts = decoder.prompts([user["items"][:position] for user, position in zip(users, positions, strict=True)])
    codes = [decoder.item_codes[user["items"][position]] for user, position in zip(users, positions, strict=True)]
    return list(zip(prompts, codes, strict=True))


def train_batch(decoder, batch, optimizer, parameters):
    prompts, codes = zip(*batch)
    optimizer.zero_grad(set_to_none=True)
    loss = decoder.training_loss(prompts, codes)
    loss.backward()
    update(optimizer, parameters)
    return loss.detach()


def benchmark_variant(args, data, indices, batch_size, *, reference):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(SEED)
    phases = {}
    if reference:
        from ..models.training_prefix import backward_prefix_all

        decision = load_decision(args, args.source_checkpoint, train=True)
        optimizer = optimizer_for(decision, decision.head.parameters())
        parameters = [param for param in decision.parameters() if param.requires_grad]
        batches = indices
        prep_seconds = 0.0
    else:
        decoder = new_decoder(args, data, args.source_checkpoint, train=True)
        decision = decoder.decision
        optimizer = optimizer_for(decision, decoder.heads.parameters())
        parameters = [param for param in decoder.parameters() if param.requires_grad]
        started = perf_counter()
        samples = prepare_train(decoder, [data.users[index] for index in indices])
        prep_seconds = perf_counter() - started
        batches = [samples[start:start + batch_size] for start in range(0, len(samples), batch_size)]
    synchronize(device)
    start = perf_counter()
    warm_batches = min(4, len(batches) - 1)
    steady_start = start
    for step, batch in enumerate(batches, 1):
        if reference:
            rows = [data.row("train", batch * data.depth + level, catalog=True) for level in range(data.depth)]
            rows = [row for row in rows if len(row["options"]) > 1]
            optimizer.zero_grad(set_to_none=True)
            if rows:
                backward_prefix_all(decision, rows, 16, 0.1, 1 / len(rows), prefix_batch_size=32,
                                    timings=phases if step <= min(warm_batches, 4) else None,
                                    audit=None, usage=None)
                update(optimizer, parameters)
        else:
            train_batch(decoder, batch, optimizer, parameters)
        if step == warm_batches:
            synchronize(device)
            steady_start = perf_counter()
        if step % 64 == 0:
            log("bench_progress", scorer="prefix-all" if reference else "sid-projection",
                batch_size=batch_size, users_done=min(step * batch_size, len(indices)), users_total=len(indices))
    synchronize(device)
    end = perf_counter()
    steady = (end - steady_start) / (len(indices) - warm_batches * batch_size)
    seconds_per_user = steady + prep_seconds / len(indices)
    return {"status": "complete", "seconds": end - start, "prepare_seconds": prep_seconds,
            "steady_seconds_per_user": steady, "seconds_per_user": seconds_per_user,
            "estimated_epoch_hours": seconds_per_user * len(data.users) / 3600,
            "linear_attention_kernels": sorted({layer.chunk_gated_delta_rule.__name__
                for layer in decision.modules() if callable(getattr(layer, "chunk_gated_delta_rule", None))}),
            "profile_warmup_seconds": phases, **memory(device)}


def benchmark_decoding(args, data):
    device = torch.device(args.device)
    decoder = new_decoder(args, data, args.source_checkpoint, train=False)
    users = sampled_users(data.users, 32)
    if args.decode_backend == "cache":
        audit_cache(decoder, audit_histories(users))
    speeds = {}
    for cutoff in (VALIDATION_K, TOP_K):
        synchronize(device)
        started = perf_counter()
        entries = list(ranking_batches(decoder, users, args.dataset, "validation", cutoff, args))
        synchronize(device)
        seconds = perf_counter() - started
        speeds[str(cutoff)] = {"users": len(entries), "seconds": seconds,
                               "seconds_per_user": seconds / len(entries),
                               "estimated_full_hours": seconds * len(data.users) / len(entries) / 3600}
    return speeds


def benchmark(args, data):
    indices = list(range(len(data.users)))
    random.Random(SEED).shuffle(indices)
    indices = indices[:args.benchmark_users]
    if max(args.benchmark_batches) > len(indices):
        raise ValueError("Benchmark user count must cover the largest requested batch")
    results = []
    variants = [(batch, False) for batch in args.benchmark_batches]
    if not args.skip_reference:
        variants.append((1, True))
    for batch, reference in variants:
        try:
            row = benchmark_variant(args, data, indices, batch, reference=reference)
        except torch.OutOfMemoryError:
            row = {"status": "oom", **memory(torch.device(args.device))}
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        row.update(batch_size=batch, scorer="prefix-all" if reference else "sid-projection", users=len(indices))
        results.append(row)
        log("bench_result", **row)
    successful = [row for row in results if row["status"] == "complete" and row["scorer"] == "sid-projection"]
    if not successful:
        write_json(args.output / "benchmark.json", {"seed": SEED, "results": results})
        raise RuntimeError("All SID projection batches ran out of memory")
    fastest = min(successful, key=lambda row: row["seconds_per_user"])
    report = {"seed": SEED, "results": results, "fastest_projection": fastest}
    try:
        report["decoding"] = benchmark_decoding(args, data)
        log("bench_decoding", estimates=report["decoding"])
    finally:
        write_json(args.output / "benchmark.json", report)


def ranking_batches(decoder, users, dataset, split, cutoff, args):
    set_training_mode(decoder, False)

    def item_order(item):
        return item_key(dataset, item)

    for start in range(0, len(users), args.decode_batch):
        batch = users[start:start + args.decode_batch]
        history, targets = zip(*(holdout(user, split) for user in batch))
        prompts = decoder.prompts(history)
        ranked = decoder.recommend(prompts, history, cutoff, item_order, cached=args.decode_backend == "cache")
        for user, target, predictions in zip(batch, targets, ranked, strict=True):
            yield ranked_entry(user["user_id"], target, predictions)
        if split == "validation" and (start // args.decode_batch + 1) % 64 == 0:
            log("validation_progress", users_done=min(start + len(batch), len(users)), users_total=len(users))


def audit_histories(valid_users):
    return [holdout(user, "validation")[0] for user in valid_users[:2]]


def audit_cache(decoder, histories):
    set_training_mode(decoder, False)
    reference = torch.tensor(decoder.audit_logits(histories, cached=False))
    cached = torch.tensor(decoder.audit_logits(histories, cached=True))
    torch.testing.assert_close(cached, reference, atol=0.06, rtol=0.03)
    log("cache_audit", max_abs_logit_difference=float((cached - reference).abs().max()))


def save_checkpoint(decoder, destination, identity, anchors, args):
    with staged_output(destination) as checkpoint:
        decoder.decision.save(checkpoint)
        config = checkpoint / "model.json"
        config.write_text(json.dumps({**read_json(config), "method": METHOD}, indent=2) + "\n", encoding="utf-8")
        torch.save(decoder.heads.state_dict(), checkpoint / "sid_projection.pt")
        values = decoder.audit_logits(anchors, cached=args.decode_backend == "cache")
        write_json(checkpoint / "sid_decoder.json", {"inputs": identity, "tokens": decoder.tokens, "audit_logits": values})


def fit(args, data, identity, train_users, valid_users):
    device = torch.device(args.device)
    decoder = new_decoder(args, data, args.source_checkpoint, train=True)
    optimizer = optimizer_for(decoder.decision, decoder.heads.parameters())
    parameters = [param for param in decoder.parameters() if param.requires_grad]
    started = perf_counter()
    samples = prepare_train(decoder, train_users)
    log("train_prompts_prepared", users=len(samples), seconds=perf_counter() - started)
    anchors = audit_histories(valid_users)
    if args.decode_backend == "cache":
        audit_cache(decoder, anchors)
    best, misses = None, 0
    stop_reason = "max_epochs"
    with (args.output / "training.jsonl").open("x", encoding="utf-8") as stream:
        for epoch in range(1, args.epochs + 1):
            set_training_mode(decoder, True)
            order = list(range(len(samples)))
            random.Random(SEED + epoch).shuffle(order)
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = perf_counter()
            loss = torch.zeros((), device=device)
            for offset in range(0, len(order), args.train_batch):
                batch = [samples[index] for index in order[offset:offset + args.train_batch]]
                loss += len(batch) * train_batch(decoder, batch, optimizer, parameters)
                if (offset // args.train_batch + 1) % 64 == 0:
                    log("train_progress", epoch=epoch, users_done=min(offset + len(batch), len(samples)),
                        users_total=len(samples))
            synchronize(device)
            record = {"epoch": epoch, "users": len(samples), "updates": math.ceil(len(samples) / args.train_batch),
                      "loss": float(loss) / len(samples), "train_seconds": perf_counter() - start, **memory(device)}
            log("train_epoch", **record)
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                started = perf_counter()
                entries = list(ranking_batches(decoder, valid_users, args.dataset, "validation", VALIDATION_K, args))
                record.update(validation_metrics(entries, valid_users), validation_seconds=perf_counter() - started)
                if best is None or record["NDCG@10"] > best["ndcg_at10"]:
                    destination = args.output / f"epoch-{epoch:04d}"
                    save_checkpoint(decoder, destination, identity, anchors, args)
                    best = {"epoch": epoch, "ndcg_at10": record["NDCG@10"],
                            "checkpoint": destination.name, "checkpoint_sha256": checkpoint_digest(destination)}
                    misses = 0
                else:
                    misses += 1
                record["validation_misses"] = misses
                log("validation", **record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if ("validation_misses" in record and args.patience and epoch < args.epochs
                    and epoch >= args.min_epochs and misses >= args.patience):
                stop_reason = "validation_patience"
                log("early_stop", epoch=epoch, validation_misses=misses, best_epoch=best["epoch"])
                break
    selection = {"best_validation": best, "validation_protocol": validation_record(valid_users, smoke=args.smoke),
                 "train_selection": TRAIN_SELECTION, "inputs": identity,
                 "epochs_trained": epoch, "stop_reason": stop_reason}
    write_json(args.output / "selection.json", selection)
    log("checkpoint_selected", epochs_trained=epoch, stop_reason=stop_reason, **best)
    return selection


def load_projection_checkpoint(args, data, path, inputs, valid_users, *, train=False):
    saved = read_json(path / "sid_decoder.json")
    if saved["inputs"] != inputs:
        raise ValueError("The selected SID checkpoint uses different recorded inputs")
    decoder = new_decoder(args, data, path, train=train)
    if saved["tokens"] != decoder.tokens:
        raise ValueError("SID output vocabulary differs from the selected checkpoint")
    decoder.heads.load_state_dict(torch.load(path / "sid_projection.pt", weights_only=True,
                                             map_location=decoder.device))
    set_training_mode(decoder, False)
    expected = torch.tensor(saved["audit_logits"])
    actual = torch.tensor(decoder.audit_logits(audit_histories(valid_users), cached=args.decode_backend == "cache"))
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    log("reload_audit", max_abs_logit_difference=float((actual - expected).abs().max()))
    set_training_mode(decoder, train)
    return decoder


def reuse_evaluation(source, final, users, catalog, cohort, details, selection):
    if not source or not source.is_file():
        return False
    try:
        result, _ = read_evaluation(source, users, catalog)
    except (OSError, ValueError) as error:
        log("prior_evaluation_unavailable", reason=str(error))
        return False
    keys = ("method", "beam_size", "protocol", "scoring_backend", "decode_batch", "checkpoint_sha256")
    expected = {**cohort, **{key: details[key] for key in keys}, "model": "OpenJev",
                "train_selection": selection["train_selection"],
                "validation_protocol": selection["validation_protocol"]}
    if any(result.get(key) != value for key, value in expected.items()):
        return False
    write_evaluation(final, read_json(source)["entries"], users, catalog, cohort, "OpenJev", **details,
                     train_selection=selection["train_selection"], validation_protocol=selection["validation_protocol"],
                     best_validation=selection["best_validation"])
    log("evaluation_reused", source=str(source), output=str(final))
    return True


def test(args, data, identity, users, cohort, selection, valid_users, *, checkpoint=None, checkpoint_inputs=None,
         extra_details=None, prior_evaluation=None):
    chosen = selection["best_validation"]
    path = checkpoint if checkpoint is not None else args.output / chosen["checkpoint"]
    if selection["inputs"] != identity or checkpoint_digest(path) != chosen["checkpoint_sha256"]:
        raise ValueError("Selected SID checkpoint differs from the recorded validation run")
    final = args.output / "evaluation.json"
    details = {"method": METHOD, "beam_size": TOP_K, "seed": SEED,
               "protocol": "full catalog, seen items excluded, autoregressive beam, no teacher forcing",
               "scoring_backend": args.decode_backend, "decode_batch": args.decode_batch,
               "checkpoint_sha256": chosen["checkpoint_sha256"]}
    if args.initialization == "published":
        details.update(initialization="published", source_checkpoint_sha256=identity["source_checkpoint_sha256"],
                       epochs_trained=selection["epochs_trained"], stop_reason=selection["stop_reason"])
    details.update(extra_details or {})
    settings = {**cohort, **details}
    if final.exists():
        result, _ = read_evaluation(final, users, data.item_ids)
        if any(result.get(key) != value for key, value in settings.items()):
            raise ValueError("The saved evaluation belongs to a different checkpoint or settings")
        log("evaluation_reused", output=str(final))
        return
    if reuse_evaluation(prior_evaluation, final, users, data.item_ids, cohort, details, selection):
        return
    specs = shard_specs(args.output / "test-shards", len(users), TEST_SHARD_SIZE, settings)
    decoder = None
    for directory, metadata in specs:
        shard_users = users[metadata["start"]:metadata["stop"]]
        if directory.exists():
            checked_ranks(read_shard(directory, metadata), shard_users, data.item_ids)
            continue
        if decoder is None:
            decoder = load_projection_checkpoint(args, data, path, checkpoint_inputs or identity, valid_users)
        started = perf_counter()
        entries = list(ranking_batches(decoder, shard_users, args.dataset, "test", TOP_K, args))
        checked_ranks(entries, shard_users, data.item_ids)
        with staged_output(directory) as stage:
            write_json(stage / "rankings.json", {"metadata": metadata, "entries": entries})
        seconds = perf_counter() - started
        log("test_progress", users_done=metadata["stop"], users_total=len(users),
            seconds=seconds, seconds_per_user=seconds / len(shard_users))
    result = merge_shards(final, specs, users, data.item_ids, cohort, details, selection)
    log("evaluation_complete", output=str(final), **result)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bench", "run", "test"))
    for name in ("sid-data", "comparison-data", "cohort", "source-checkpoint", "base-model", "output"):
        parser.add_argument(f"--{name}", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--initialization", choices=("sft", "published"), default="sft",
                        help="Load completed SID SFT or initialize SID training from released OpenJev")
    parser.add_argument("--dataset", choices=(*FORMAL_DATASETS, "ml100k"), default="ml1m")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Use eight users per split for a local check")
    parser.add_argument("--benchmark-users", type=int, default=512)
    parser.add_argument("--benchmark-batches", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--train-batch", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--min-epochs", type=int, help="Earliest epoch that permits early stopping")
    parser.add_argument("--patience", type=int, help="Validation checks without improvement, or 0 to disable")
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--decode-backend", choices=("cache", "recompute"), default="cache")
    parser.add_argument("--gradient-checkpointing", choices=("on", "partial", "off"), default="partial")
    args = parser.parse_args()
    if args.dataset == "ml100k" and not args.smoke:
        parser.error("ML-100K is available for local smoke checks")
    if args.stage == "run":
        published_formal = args.initialization == "published" and not args.smoke
        if args.epochs is None and args.initialization == "published":
            args.epochs = PUBLISHED_EPOCHS if published_formal else 1
        if args.min_epochs is None:
            args.min_epochs = PUBLISHED_MIN_EPOCHS if published_formal else 1
        if args.patience is None:
            args.patience = PUBLISHED_PATIENCE if published_formal else 0
        if args.epochs is None or not 1 <= args.min_epochs <= args.epochs or args.patience < 0:
            parser.error("Training needs 1 <= min-epochs <= epochs and nonnegative patience")
        if published_formal and (args.min_epochs < PUBLISHED_MIN_EPOCHS or args.validate_every != 1):
            parser.error("Published formal training needs at least 5 epochs and validation every epoch")
    if min(args.train_batch, args.decode_batch, args.validate_every, args.benchmark_users,
           *args.benchmark_batches) < 1:
        parser.error("User counts, batch sizes and validation interval must be positive")
    return args


def source_inputs(args, data):
    checkpoint = args.source_checkpoint
    sid_path = args.sid_data / "item_semantic_ids.jsonl"
    if args.initialization == "sft":
        require_sft_source(checkpoint, data)
        config = checkpoint_config(checkpoint, sid_path)
        if config["max_history"] != MAX_HISTORY:
            raise ValueError("The source SFT history window differs from the comparison protocol")
        return {"sid_sha256": config["semantic_ids_sha256"], "source_sft_sha256": checkpoint_digest(checkpoint)}

    config = read_json(checkpoint / "model.json")
    if (config.get("method") != "independent_candidate_lora_nll_brier"
            or config.get("sid_encoding", "text") != "text"
            or config.get("history_memory") or (checkpoint / "sid_tokens.json").exists()
            or (checkpoint / "history_memory.json").exists()):
        raise ValueError("Published initialization needs the original OpenJev checkpoint before SID SFT")
    for name in ("head.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Missing published OpenJev checkpoint file: {checkpoint / name}")
    sid_hash = sha256(sid_path)
    if sid_hash != data.manifest["semantic_ids_sha256"]:
        raise ValueError("The SID mapping differs from the prepared data manifest")
    return {"initialization": "published", "sid_sha256": sid_hash,
            "source_checkpoint_sha256": checkpoint_digest(checkpoint)}


def execute(args):
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or device.index not in (None, 0)):
        raise ValueError("Expose the assigned GPU as cuda:0")
    data = SequenceData(args.sid_data, args.comparison_data)
    if data.comparison_identity["dataset"] != args.dataset:
        raise ValueError("Selected dataset differs from the SID data")
    source = source_inputs(args, data)
    all_test_users, cohort = load_cohort(args.cohort, args.dataset, data.users, data.item_ids, MAX_HISTORY)
    if not args.smoke:
        require_formal_cohort(args.dataset, data, all_test_users, cohort)
    identity = {"method": METHOD, "cohort": cohort, "comparison_data": data.comparison_identity,
                **source,
                "base_sha256": encoder_fingerprint(args.base_model), "smoke": args.smoke, **runtime_identity(args)}
    if args.stage != "test":
        write_json(args.output / "run.json", {"inputs": identity, "train_selection": TRAIN_SELECTION,
                   "arguments": {name: str(value) if isinstance(value, Path) else value
                                 for name, value in vars(args).items()}})
    if args.stage == "bench":
        benchmark(args, data)
        return
    if args.smoke:
        train_users = valid_users = sampled_users(data.users, 8)
        users = sampled_users(all_test_users, 8)
        cohort = {**cohort, "scope": "smoke", "cohort_sha256": hashlib.sha256(json.dumps(
            [user["user_id"] for user in users]).encode()).hexdigest()}
    else:
        train_users = valid_users = data.users
        users = all_test_users
    if args.stage == "run":
        selection = fit(args, data, identity, train_users, valid_users)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        selection = read_json(args.output / "selection.json")
    if selection["validation_protocol"] != validation_record(valid_users, smoke=args.smoke):
        raise ValueError("Selected checkpoint did not use the required validation cohort")
    test(args, data, identity, users, cohort, selection, valid_users)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        execute(args)


if __name__ == "__main__":
    main()
