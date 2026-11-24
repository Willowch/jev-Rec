"""Compute the comparison table from saved Top-20 rankings."""

import argparse
from pathlib import Path

from . import protocol
from .prepare import load_prepared
from .protocol import (MAX_HISTORY, METRIC_CUTOFFS, TOP_K, load_cohort,
                       ranking_metrics, read_evaluation, require_training_selection, validation_record,
                       validation_users)


BASELINES = ("TIGER", "SASRec", "BERT4Rec", "LETTER-TIGER")
MODELS = ("OpenJev", *BASELINES)


def summarize(paths, data_dir, cohort_path, dataset, include_at20, *, baselines_only=False):
    users, items, _ = load_prepared(data_dir, dataset)
    selected, metadata = load_cohort(cohort_path, dataset, users, items, MAX_HISTORY)
    models = BASELINES if baselines_only else MODELS
    pilot = dataset == getattr(protocol, "PILOT_DATASET", None)
    if metadata["scope"] != ("pilot" if pilot else "formal"):
        raise ValueError("the comparison table requires the matching pilot or formal cohort")
    cutoffs = METRIC_CUTOFFS if include_at20 or baselines_only else METRIC_CUTOFFS[:-1]
    expected_validation = validation_record(validation_users(users, pilot=True) if pilot else users)
    results = {}
    display_names = {}
    for path in paths:
        result, ranks = read_evaluation(path, selected, items)
        model = result["model"]
        if model not in models or model in results:
            raise ValueError(f"{path}: unknown or repeated model {model}")
        if any(result.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"{path}: result was generated with a different cohort or cutoff")
        if result["users"] != len(selected):
            raise ValueError(f"{path}: reported user count differs from the cohort")
        full_catalog_openjev = model == "OpenJev" and result.get("ranking_mode") == "full_catalog"
        if full_catalog_openjev:
            if result.get("method") != "openjev_full_catalog_v1" or result.get("catalog_size") != len(items):
                raise ValueError(f"{path}: full-catalog OpenJev must score the entire shared catalog")
            display_names[model] = "OpenJev (full catalog)"
        elif model in ("OpenJev", "TIGER", "LETTER-TIGER") and result.get("beam_size") != TOP_K:
            raise ValueError(f"{path}: expected beam size {TOP_K}")
        require_training_selection(result)
        if result.get("validation_protocol") != expected_validation:
            raise ValueError(f"{path}: {model} must use the shared validation set for NDCG@10 selection")
        results[model] = ranking_metrics(ranks, cutoffs)
    if results.keys() != set(models):
        missing = [model for model in models if model not in results]
        raise ValueError(f"missing evaluation results for {', '.join(missing)}")

    columns = ["Dataset", "Model", "Users", *results[models[0]]]
    lines = [columns, ["---"] * len(columns)]
    for model in models:
        values = [str(number) if isinstance(number, int) else f"{number:.6f}"
                  for number in results[model].values()]
        lines.append([dataset, display_names.get(model, model), str(len(selected)), *values])
    return "\n".join("| " + " | ".join(row) + " |" for row in lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=getattr(protocol, "COMPARISON_DATASETS", protocol.DATASETS),
                        required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--include-at20", action="store_true")
    parser.add_argument("--baselines-only", action="store_true", help="Summarize TIGER, SASRec, BERT4Rec and LETTER-TIGER")
    parser.add_argument("--output", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    if args.baselines_only:
        if len(args.results) != len(BASELINES):
            parser.error("pass the four baseline evaluation files")
    elif len(args.results) != len(MODELS):
        parser.error(f"pass five evaluation files ({', '.join(MODELS)})")
    table = summarize(args.results, args.data_dir, args.cohort, args.dataset, args.include_at20,
                      baselines_only=args.baselines_only)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(table)
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
