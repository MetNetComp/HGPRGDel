"""Manuscript Table VI/VII metabolite-based five-fold partitions (seed 42)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def partitions_path(root: Path | None = None) -> Path:
    """Canonical fold file at ``utils/cv_partitions.json`` (shipped in the public ZIP)."""
    return (root or project_root()) / "utils" / "cv_partitions.json"


def load_gem_partitions(cbm: str, root: Path | None = None) -> dict[str, Any]:
    path = partitions_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Expected the public-release file "
            "utils/cv_partitions.json under the repository root."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if cbm not in data:
        raise KeyError(f"Unknown GEM {cbm!r} in {path}")
    return data[cbm]


def all_target_metabolites(cbm: str, root: Path | None = None) -> list[str]:
    spec = load_gem_partitions(cbm, root)
    seen: set[str] = set()
    for fold in spec["folds"]:
        seen.update(fold["val_metabolites"])
    return sorted(seen)


def train_val_metabolites(
    cbm: str,
    fold_index: int,
    root: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return (train_metabolites, val_metabolites) for fold_index in ``0 .. num_folds-1``."""
    spec = load_gem_partitions(cbm, root)
    folds = spec["folds"]
    if fold_index < 0 or fold_index >= len(folds):
        raise IndexError(f"fold_index {fold_index} out of range for {cbm} ({len(folds)} folds)")
    val_metabolites = list(folds[fold_index]["val_metabolites"])
    val_set = set(val_metabolites)
    train_metabolites = [m for m in all_target_metabolites(cbm, root) if m not in val_set]
    return train_metabolites, val_metabolites
