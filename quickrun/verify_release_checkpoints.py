#!/usr/bin/env python3
"""Verify quick-run checkpoint files shipped in this GitHub repository."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# HGPRGDel default quick_run discs (all three GEMs) + e_coli_core baselines.
RELEASE_FILES = [
    "utils/cv_partitions.json",
    "checkpoints/model_e_coli_core.sav",
    "checkpoints/model_iMM904.sav",
    "checkpoints/model_iML1515.sav",
    "baseline/GraphGdel_e_coli_core.sav",
    "baseline/DNN_e_coli_core_Gene_AA.sav",
    "baseline/DeepGdel_e_coli_core.sav",
    "baseline/NMA_e_coli_core.sav",
]


def main() -> int:
    missing: list[str] = []
    for rel in RELEASE_FILES:
        path = ROOT / rel
        if path.is_file():
            print(f"OK  {rel}")
            continue
        print(f"MISSING  {rel}")
        missing.append(rel)

    if missing:
        print(
            "\nRelease incomplete for quick_run scripts. "
            "Expected files are shipped in GitHub under checkpoints/ and baseline/. "
            "See checkpoints/README.md and baseline/README_CHECKPOINTS.md.",
            file=sys.stderr,
        )
        return 1
    print("\nAll quick-run checkpoints present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
