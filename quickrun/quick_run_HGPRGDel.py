#!/usr/bin/env python3
"""
Quick run script for HGPRGDel (default GEM: e_coli_core).

Delegates to HGPRGDel_main.main with default ``--CBM e_coli_core`` and ``--use_cpu 1``
(mapped to ``--device`` in ``utils.hgprgdel_quick_run``). Additional CLI arguments are
forwarded; a later ``--CBM`` overrides the default (supported: e_coli_core, iML1515, iMM904).

**Outputs:** default ``--experiment_dir`` is ``runs/quick_run/<CBM>/`` (under the repo root).

**Weights:** when not using ``--train``, if ``--resume_checkpoint`` is omitted, quick run
loads **only** from ``checkpoints/``: ``checkpoints/model_<CBM>.sav`` (PyTorch state dict),
or ``checkpoints/<CBM>/`` (``best_overall.sav`` / fold ``checkpoints/``). Pass
``--resume_checkpoint PATH`` for any other checkpoint.

Flag ``--train`` is stripped before ``HGPRGDel_main`` and triggers full CV without
requiring a prior checkpoint. See ``utils/hgprgdel_quick_run.py`` for argv handling.

After checkpoint inference (not ``--train``), prints overall evaluable-pair metrics
(accuracy, macro P/R/F1, AUC) to stdout using the default train-macro-F1 global threshold
(tuned on manuscript fold-0 training metabolites unless ``--train`` wrote fold splits).
Pass ``--binary_threshold youden_per_metabolite`` for the legacy per-metabolite ROC protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from HGPRGDel_main import main as hgprgdel_main
from utils.data_module import PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD
from utils.hgprgdel_quick_run import apply_hgprgdel_quick_run_argv
from utils.paths import project_root
from utils.prediction_threshold import DEFAULT_BINARY_THRESHOLD_MODE, BinaryThresholdMode


def _last_argv_value(flag: str, default: str) -> str:
    value = default
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            value = argv[i + 1]
    return value


def _print_quick_run_overall_metrics() -> None:
    """Print overall evaluable-pair metrics from the prediction CSV (stdout only)."""
    cbm = _last_argv_value("--CBM", "e_coli_core")
    root = project_root()
    default_exp = str(root / "runs" / "quick_run" / cbm)
    exp_dir = Path(_last_argv_value("--experiment_dir", default_exp))
    pred_path = exp_dir / "predictions" / "all_metabolites_predictions.csv"
    if not pred_path.is_file():
        print(f"Quick-run metrics: skipped (no predictions at {pred_path})", file=sys.stderr)
        return

    eval_threshold = float(_last_argv_value(
        "--eval_threshold",
        str(PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD),
    ))
    binary_threshold: BinaryThresholdMode = _last_argv_value(
        "--binary_threshold",
        DEFAULT_BINARY_THRESHOLD_MODE,
    )  # type: ignore[assignment]

    from utils.summarize_metrics import summarize_run

    summary = summarize_run(
        exp_dir,
        eval_threshold=eval_threshold,
        binary_threshold=binary_threshold,
        cbm=cbm,
    )
    overall = summary["overall"]
    auc = overall.get("auc")
    auc_str = f"{auc:.4f}" if auc is not None else "n/a"

    print("\n=== Quick-run overall metrics (evaluable pairs, full GEM, single checkpoint) ===")
    print(
        f"  GEM: {cbm}  |  n_pairs_total: {summary['n_pairs_total']}"
        f"  |  n_evaluable: {summary['n_pairs_evaluated']}"
        f"  |  eval_threshold: {eval_threshold}"
        f"  |  binary_threshold: {binary_threshold}"
    )
    print(
        f"  accuracy: {overall['accuracy']:.4f}"
        f"  |  precision_macro: {overall['precision_macro']:.4f}"
        f"  |  recall_macro: {overall['recall_macro']:.4f}"
        f"  |  f1_macro: {overall['f1_macro']:.4f}"
        f"  |  auc: {auc_str}"
    )


def run() -> None:
    prog = sys.argv[0] if sys.argv else "quick_run_HGPRGDel.py"
    want_train = "--train" in sys.argv
    if len(sys.argv) <= 1:
        sys.argv = [prog, "--CBM", "e_coli_core", "--use_cpu", "1"]
    else:
        sys.argv = [prog, "--CBM", "e_coli_core", "--use_cpu", "1"] + sys.argv[1:]
    apply_hgprgdel_quick_run_argv()
    hgprgdel_main()
    if not want_train:
        _print_quick_run_overall_metrics()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
