#!/usr/bin/env python3
"""Summarize HGPRGDel prediction CSV files for manuscript tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Optional

from utils.data_module import (
    PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD,
    get_predetermined_genes_from_prediction_rows,
)
from utils.manuscript_cv import train_val_metabolites
from utils.prediction_threshold import (
    BINARY_THRESHOLD_TRAIN_MACRO_F1,
    BINARY_THRESHOLD_YOUDEN,
    DEFAULT_BINARY_THRESHOLD_MODE,
    BinaryThresholdMode,
    classification_metrics_from_rows,
    global_threshold_from_train_rows,
    threshold_mode_description,
)


METRIC_KEYS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "auc")


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return sum(clean) / len(clean) if clean else None


def _std(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if len(clean) < 2:
        return 0.0 if clean else None
    return statistics.stdev(clean)


def _rows_for_evaluation(
    rows: list[dict[str, str]],
    eval_threshold: float = PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD,
) -> list[dict[str, str]]:
    """Drop pre-determined genes; keep all pairs for inference export otherwise."""
    predetermined = get_predetermined_genes_from_prediction_rows(rows, threshold=eval_threshold)
    if not predetermined:
        return rows
    return [row for row in rows if row["Gene"] not in predetermined]


def _load_train_metabolites_for_run(
    run_dir: Path,
    cbm: Optional[str],
) -> Optional[list[str]]:
    train_paths = sorted((run_dir / "splits").glob("fold_*_train_metabolites.json"))
    if train_paths:
        return list(json.loads(train_paths[0].read_text()))
    if cbm:
        try:
            train_metabolites, _ = train_val_metabolites(cbm, 0)
            return train_metabolites
        except FileNotFoundError:
            return None
    return None


def summarize_run(
    run_dir: Path,
    eval_threshold: float = PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD,
    binary_threshold: BinaryThresholdMode = DEFAULT_BINARY_THRESHOLD_MODE,
    cbm: Optional[str] = None,
) -> dict[str, object]:
    prediction_path = run_dir / "predictions" / "all_metabolites_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

    with prediction_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eval_rows = _rows_for_evaluation(rows, eval_threshold=eval_threshold)
    train_metabolites = _load_train_metabolites_for_run(run_dir, cbm)
    if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1 and not train_metabolites:
        # Quick-run public ZIP may omit CV partitions; tune on all metabolites in the CSV.
        train_metabolites = sorted({row["Metabolite"] for row in eval_rows})
    global_threshold = None
    if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1 and train_metabolites:
        global_threshold = global_threshold_from_train_rows(eval_rows, train_metabolites)

    fold_results: list[dict[str, object]] = []
    for split_path in sorted((run_dir / "splits").glob("fold_*_val_metabolites.json")):
        fold = int(split_path.stem.split("_")[1])
        val_metabolites = set(json.loads(split_path.read_text()))
        train_path = run_dir / "splits" / f"fold_{fold}_train_metabolites.json"
        fold_train_metabolites = (
            json.loads(train_path.read_text()) if train_path.is_file() else train_metabolites
        )
        fold_rows = [row for row in eval_rows if row["Metabolite"] in val_metabolites]
        fold_threshold = global_threshold
        if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1 and fold_train_metabolites:
            fold_threshold = global_threshold_from_train_rows(eval_rows, fold_train_metabolites)
        metrics = classification_metrics_from_rows(
            fold_rows,
            mode=binary_threshold,
            train_metabolites=fold_train_metabolites,
            global_threshold=fold_threshold,
        )
        fold_metrics = {key: metrics[key] for key in METRIC_KEYS}
        fold_results.append(
            {
                "fold": fold + 1,
                "n": len(fold_rows),
                "binary_threshold": metrics.get("binary_threshold"),
                **fold_metrics,
            }
        )

    train_metabolites = _load_train_metabolites_for_run(run_dir, cbm)
    overall = classification_metrics_from_rows(
        eval_rows,
        mode=binary_threshold,
        train_metabolites=train_metabolites,
        global_threshold=global_threshold,
    )
    overall_metrics = {key: overall[key] for key in METRIC_KEYS}
    summary = {
        "run_dir": str(run_dir),
        "source": "predictions/all_metabolites_predictions.csv",
        "eval_threshold": eval_threshold,
        "binary_threshold": binary_threshold,
        "n_pairs_total": len(rows),
        "n_pairs_evaluated": len(eval_rows),
        "note": (
            "Fold summaries group the saved all-metabolite prediction file with "
            "fold validation metabolite lists and exclude pre-determined genes "
            f"(>={eval_threshold:.0%} label consistency) from metrics. "
            f"Binary predictions use {threshold_mode_description(binary_threshold)}."
        ),
        "overall": {
            "n": len(eval_rows),
            "binary_threshold": overall.get("binary_threshold"),
            **overall_metrics,
        },
        "fold_results": fold_results,
        "mean": {key: _mean(fold[key] for fold in fold_results) for key in METRIC_KEYS},
        "std": {key: _std(fold[key] for fold in fold_results) for key in METRIC_KEYS},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD,
        help="Exclude genes with >= this label-consistency fraction from metrics",
    )
    parser.add_argument(
        "--binary-threshold",
        choices=[BINARY_THRESHOLD_TRAIN_MACRO_F1, BINARY_THRESHOLD_YOUDEN],
        default=DEFAULT_BINARY_THRESHOLD_MODE,
        help="How to binarize scores when summarizing metrics",
    )
    parser.add_argument("--cbm", type=str, default=None, help="GEM for default train-metabolite split when splits/ is missing")
    args = parser.parse_args()

    summaries = {
        run_dir.name: summarize_run(
            run_dir,
            eval_threshold=args.eval_threshold,
            binary_threshold=args.binary_threshold,
            cbm=args.cbm,
        )
        for run_dir in args.run_dirs
    }
    payload = {"runs": summaries}
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
