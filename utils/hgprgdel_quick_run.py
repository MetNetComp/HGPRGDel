"""Argv adapters and default path injection for ``quickrun/quick_run_HGPRGDel.py`` before ``HGPRGDel_main.main``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from utils.paths import (
    default_experiment_config,
    project_root,
    resolve_quick_run_checkpoint,
)


def _normalize_use_cpu(argv: list[str]) -> list[str]:
    """Map ``--use_cpu 0|1`` to ``--device auto|cpu`` (last pair wins)."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--use_cpu" and i + 1 < len(argv):
            out.extend(["--device", "cpu" if argv[i + 1] == "1" else "auto"])
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _strip_train_flag(argv: list[str]) -> tuple[list[str], bool]:
    out: list[str] = []
    want_train = False
    i = 0
    while i < len(argv):
        if argv[i] == "--train":
            want_train = True
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out, want_train


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _last_value_after(argv: list[str], flag: str, default: str) -> str:
    v = default
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            v = argv[i + 1]
    return v


def _inject_quick_run_defaults(
    tail: list[str], *, root: Path, cbm: str, want_train: bool
) -> list[str]:
    t = list(tail)
    if not _argv_has_flag(t, "--data_root"):
        t.extend(["--data_root", str(root / "Data")])
    if not _argv_has_flag(t, "--experiment_dir"):
        exp = root / "runs" / "quick_run" / cbm
        exp.mkdir(parents=True, exist_ok=True)
        t.extend(["--experiment_dir", str(exp)])
    if not _argv_has_flag(t, "--config"):
        cfg = default_experiment_config(cbm)
        if cfg is not None:
            t.extend(["--config", str(cfg)])
    if not want_train and not _argv_has_flag(t, "--resume_checkpoint"):
        ck = resolve_quick_run_checkpoint(cbm)
        if ck is None:
            print(
                f"No checkpoint under checkpoints/ for {cbm!r}: expected "
                f"checkpoints/model_{cbm}.sav or checkpoints/{cbm}/ "
                f"(best_overall.sav / checkpoints/). "
                "Place weights under checkpoints/ (shipped in GitHub for the three GEMs), "
                "pass --resume_checkpoint PATH to any .sav, "
                "or pass --train for full CV.",
                file=sys.stderr,
            )
            sys.exit(2)
        t.extend(["--resume_checkpoint", str(ck)])
    return t


def apply_hgprgdel_quick_run_argv() -> None:
    """Call after ``run()`` set default ``--CBM`` / ``--use_cpu``; rewrite ``sys.argv`` and chdir to repo root."""
    from HGPRGDel_main import SUPPORTED_GEMS

    root = project_root()
    raw = list(sys.argv)
    body, want_train = _strip_train_flag(raw[1:])
    body = _normalize_use_cpu(body)
    cbm = _last_value_after(body, "--CBM", "e_coli_core")
    if cbm not in SUPPORTED_GEMS:
        print(
            f"Invalid --CBM {cbm!r}. Choose one of: {', '.join(SUPPORTED_GEMS)}.",
            file=sys.stderr,
        )
        sys.exit(2)
    body = _inject_quick_run_defaults(body, root=root, cbm=cbm, want_train=want_train)

    os.chdir(root)
    sys.argv = ["HGPRGDel_main.py"] + body
