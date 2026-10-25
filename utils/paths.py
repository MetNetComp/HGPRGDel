"""Repository-relative paths shared by helper scripts (quick run, jobs, etc.)."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def manuscript_config_dir() -> Path:
    """Author-only manuscript YAMLs (not in public ZIP); optional local path."""
    return project_root() / "configs" / "manuscript"


def default_experiment_config(cbm: str) -> Path | None:
    """Training YAML under ``configs/manuscript/<CBM>.yaml`` if present locally."""
    p = manuscript_config_dir() / f"{cbm}.yaml"
    if p.is_file():
        return p
    p2 = per_gem_checkpoint_dir(cbm) / "config.yaml"
    return p2 if p2.is_file() else None


def manuscript_experiment_dir(cbm: str) -> Path:
    return project_root() / "runs" / "manuscript" / cbm


def checkpoints_root() -> Path:
    """Released and local HGPRGDel weights under ``checkpoints/``."""
    return project_root() / "checkpoints"


def per_gem_checkpoint_dir(cbm: str) -> Path:
    """Per-GEM checkpoint subtree: ``checkpoints/<CBM>/``."""
    return checkpoints_root() / cbm


CHECKPOINT_EXT = ".sav"
LEGACY_CHECKPOINT_EXT = ".pth"


def flat_checkpoint_path(cbm: str) -> Path:
    """Quick-run weights at ``checkpoints/model_<CBM>.sav`` (PyTorch state dict)."""
    return checkpoints_root() / f"model_{cbm}{CHECKPOINT_EXT}"


def checkpoint_filename(stem: str) -> str:
    """Standard HGPRGDel checkpoint basename, e.g. ``fold_0_best.sav``."""
    return f"{stem}{CHECKPOINT_EXT}"


def resolve_checkpoint_path(directory: Path, stem: str) -> Path | None:
    """Resolve ``<stem>.sav``, falling back to legacy ``<stem>.pth``."""
    for ext in (CHECKPOINT_EXT, LEGACY_CHECKPOINT_EXT):
        path = directory / f"{stem}{ext}"
        if path.is_file():
            return path
    return None


def _checkpoint_stems_preference() -> list[str]:
    return [
        "best_overall",
        "full_train_best",
        "fold_0_best",
        "fold_1_best",
        "fold_2_best",
        "fold_3_best",
        "fold_4_best",
    ]


def _collect_checkpoint_files(base: Path) -> list[Path]:
    found: list[Path] = []
    for stem in _checkpoint_stems_preference():
        path = resolve_checkpoint_path(base, stem)
        if path is not None:
            found.append(path)
    return found


def glob_fold_best_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """``fold_*_best.sav`` (and legacy ``.pth``) under a checkpoint directory."""
    paths = list(checkpoint_dir.glob(f"fold_*_best{CHECKPOINT_EXT}"))
    paths.extend(checkpoint_dir.glob(f"fold_*_best{LEGACY_CHECKPOINT_EXT}"))
    return sorted(paths, key=lambda p: p.name)


def list_checkpoint_candidates(cbm: str) -> list[Path]:
    """``checkpoints/model_<CBM>.sav``, else files under ``checkpoints/<CBM>/``."""
    flat = flat_checkpoint_path(cbm)
    if flat.is_file():
        return [flat]
    root = per_gem_checkpoint_dir(cbm)
    per_gem_flat = _collect_checkpoint_files(root)
    if per_gem_flat:
        return per_gem_flat
    fold_dir = root / "checkpoints"
    if fold_dir.is_dir():
        return _collect_checkpoint_files(fold_dir)
    return []


def default_checkpoint_candidates(cbm: str) -> list[Path]:
    """Released weights under ``checkpoints/``."""
    return list_checkpoint_candidates(cbm)


def resolve_default_checkpoint(cbm: str) -> Path | None:
    found = default_checkpoint_candidates(cbm)
    return found[0] if found else None


def resolve_quick_run_checkpoint(cbm: str) -> Path | None:
    """Same as :func:`resolve_default_checkpoint`."""
    return resolve_default_checkpoint(cbm)
