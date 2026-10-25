"""Binary classification thresholds for HGPRGDel and baseline prediction/metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

THRESHOLD_GRID: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.1, 0.9, 81))

BinaryThresholdMode = Literal["train_macro_f1", "youden_per_metabolite"]

BINARY_THRESHOLD_TRAIN_MACRO_F1: BinaryThresholdMode = "train_macro_f1"
BINARY_THRESHOLD_YOUDEN: BinaryThresholdMode = "youden_per_metabolite"

DEFAULT_BINARY_THRESHOLD_MODE: BinaryThresholdMode = BINARY_THRESHOLD_TRAIN_MACRO_F1


def select_global_threshold_macro_f1(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float:
    """Pick a global threshold on training scores that maximizes macro-F1."""
    label_list = [int(x) for x in labels]
    score_list = [float(x) for x in scores]
    if len(label_list) < 2 or len(set(label_list)) < 2:
        return 0.5
    best_t = THRESHOLD_GRID[0]
    best_f1 = -1.0
    for threshold in THRESHOLD_GRID:
        preds = [1 if score > threshold else 0 for score in score_list]
        f1 = f1_score(label_list, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = threshold
    return float(best_t)


def youden_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Per-metabolite ROC Youden threshold (optional legacy protocol)."""
    label_list = [int(x) for x in labels]
    score_list = [float(x) for x in scores]
    if len(label_list) < 2 or len(set(label_list)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(label_list, score_list)
    optimal_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[optimal_idx])


def _evaluable_row(row: dict[str, str]) -> bool:
    try:
        return int(row["True_Label"]) >= 0
    except (KeyError, TypeError, ValueError):
        return False


def _filter_rows(
    rows: list[dict[str, str]],
    *,
    metabolites: Optional[Iterable[str]] = None,
    evaluable_only: bool = True,
) -> list[dict[str, str]]:
    meta_set = set(metabolites) if metabolites is not None else None
    out: list[dict[str, str]] = []
    for row in rows:
        if evaluable_only and not _evaluable_row(row):
            continue
        if meta_set is not None and row["Metabolite"] not in meta_set:
            continue
        out.append(row)
    return out


def global_threshold_from_train_rows(
    rows: list[dict[str, str]],
    train_metabolites: Iterable[str],
) -> float:
    tuning_rows = _filter_rows(rows, metabolites=train_metabolites, evaluable_only=True)
    labels = [int(row["True_Label"]) for row in tuning_rows]
    scores = [float(row["Prediction"]) for row in tuning_rows]
    return select_global_threshold_macro_f1(labels, scores)


def scores_to_binary(
    scores: Sequence[float],
    threshold: float,
) -> list[int]:
    return [1 if float(score) > threshold else 0 for score in scores]


def binary_predictions_for_pairs(
    pairs: list[tuple[str, str]],
    scores: list[float],
    true_labels: dict[tuple[str, str], int],
    mode: BinaryThresholdMode = DEFAULT_BINARY_THRESHOLD_MODE,
    train_metabolites: Optional[Iterable[str]] = None,
    global_threshold: Optional[float] = None,
) -> list[int]:
    """Assign binary predictions for gene--metabolite pairs."""
    rows = [
        {
            "Gene": gene,
            "Metabolite": metabolite,
            "Prediction": scores[index],
            "True_Label": true_labels.get((gene, metabolite), -1),
        }
        for index, (gene, metabolite) in enumerate(pairs)
    ]
    binary_by_index = {
        index: pred
        for index, pred in enumerate(
            binary_predictions_from_rows(
                rows,
                mode=mode,
                train_metabolites=train_metabolites,
                global_threshold=global_threshold,
            )
        )
    }
    return [binary_by_index[index] for index in range(len(pairs))]


def binary_predictions_from_rows(
    rows: list[dict[str, str]],
    mode: BinaryThresholdMode = DEFAULT_BINARY_THRESHOLD_MODE,
    train_metabolites: Optional[Iterable[str]] = None,
    global_threshold: Optional[float] = None,
) -> list[int]:
    if mode == BINARY_THRESHOLD_TRAIN_MACRO_F1:
        if global_threshold is None:
            if train_metabolites is None:
                raise ValueError("train_metabolites required for train_macro_f1 thresholding")
            global_threshold = global_threshold_from_train_rows(rows, train_metabolites)
        return scores_to_binary([float(row["Prediction"]) for row in rows], global_threshold)

    by_metabolite: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_metabolite[row["Metabolite"]].append(index)

    thresholds: dict[str, float] = {}
    for metabolite, indices in by_metabolite.items():
        labels: list[int] = []
        meta_scores: list[float] = []
        for index in indices:
            if not _evaluable_row(rows[index]):
                continue
            labels.append(int(rows[index]["True_Label"]))
            meta_scores.append(float(rows[index]["Prediction"]))
        thresholds[metabolite] = youden_threshold(labels, meta_scores)

    binary: list[int] = []
    for index, row in enumerate(rows):
        threshold = thresholds.get(row["Metabolite"], 0.5)
        binary.append(1 if float(row["Prediction"]) > threshold else 0)
    return binary


def _per_class_macro_metrics(
    labels: list[int],
    predictions: list[int],
) -> tuple[float, float, float, float]:
    correct = sum(1 for label, pred in zip(labels, predictions) if label == pred)
    accuracy = correct / len(labels) if labels else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in (0, 1):
        tp = sum(1 for label, pred in zip(labels, predictions) if label == cls and pred == cls)
        fp = sum(1 for label, pred in zip(labels, predictions) if label != cls and pred == cls)
        fn = sum(1 for label, pred in zip(labels, predictions) if label == cls and pred != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return (
        accuracy,
        sum(precisions) / 2,
        sum(recalls) / 2,
        sum(f1s) / 2,
    )


def classification_metrics_from_rows(
    rows: list[dict[str, str]],
    mode: BinaryThresholdMode = DEFAULT_BINARY_THRESHOLD_MODE,
    train_metabolites: Optional[Iterable[str]] = None,
    global_threshold: Optional[float] = None,
) -> dict[str, float | None]:
    """Aggregate metrics over evaluable rows in *rows* (the evaluation slice)."""
    eval_rows = _filter_rows(rows, evaluable_only=True)
    if not eval_rows:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "auc": None,
            "binary_threshold": global_threshold,
        }

    if mode == BINARY_THRESHOLD_TRAIN_MACRO_F1:
        threshold = global_threshold
        if threshold is None:
            if train_metabolites is None:
                raise ValueError("train_metabolites required for train_macro_f1 metrics")
            threshold = global_threshold_from_train_rows(rows, train_metabolites)
        labels = [int(row["True_Label"]) for row in eval_rows]
        scores = [float(row["Prediction"]) for row in eval_rows]
        predictions = scores_to_binary(scores, threshold)
        accuracy, precision_macro, recall_macro, f1_macro = _per_class_macro_metrics(labels, predictions)
        try:
            auc = float(roc_auc_score(labels, scores))
        except ValueError:
            auc = None
        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "auc": auc,
            "binary_threshold": threshold,
        }

    by_metabolite: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eval_rows:
        by_metabolite[row["Metabolite"]].append(row)

    metabolite_metrics: list[dict[str, float | None]] = []
    total_correct = 0
    total_genes = 0

    for meta_rows in by_metabolite.values():
        labels = [int(row["True_Label"]) for row in meta_rows]
        scores = [float(row["Prediction"]) for row in meta_rows]
        threshold = youden_threshold(labels, scores)
        predictions = scores_to_binary(scores, threshold)
        accuracy, precision_macro, recall_macro, f1_macro = _per_class_macro_metrics(labels, predictions)
        total_correct += sum(1 for label, pred in zip(labels, predictions) if label == pred)
        total_genes += len(labels)
        try:
            auc = float(roc_auc_score(labels, scores))
        except ValueError:
            auc = None
        metabolite_metrics.append(
            {
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "auc": auc,
            }
        )

    auc_values = [m["auc"] for m in metabolite_metrics if m["auc"] is not None]
    return {
        "accuracy": total_correct / total_genes if total_genes else 0.0,
        "precision_macro": float(np.mean([m["precision_macro"] for m in metabolite_metrics])),
        "recall_macro": float(np.mean([m["recall_macro"] for m in metabolite_metrics])),
        "f1_macro": float(np.mean([m["f1_macro"] for m in metabolite_metrics])),
        "auc": float(np.mean(auc_values)) if auc_values else None,
        "binary_threshold": None,
    }


def threshold_mode_description(mode: BinaryThresholdMode) -> str:
    if mode == BINARY_THRESHOLD_YOUDEN:
        return "per-metabolite ROC Youden thresholds (optional legacy protocol)"
    return (
        "global threshold tuned on training-metabolite evaluable pairs to maximize macro-F1 "
        f"(grid {THRESHOLD_GRID[0]:.1f}..{THRESHOLD_GRID[-1]:.1f}, {len(THRESHOLD_GRID)} steps)"
    )
