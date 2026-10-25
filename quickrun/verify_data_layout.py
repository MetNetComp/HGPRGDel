#!/usr/bin/env python3
"""Verify minimal Data/ layout after unpacking the shipped ZIP archives."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"

GEMS = ("e_coli_core", "iMM904", "iML1515")
QUICK_START_GEM = "e_coli_core"

REQUIRED_FILES = ("{gem}.json", "Target metabolites_smile.csv")
EXTRACTED_DIRS = ("Gene_AA", "Label")


def _check_gem(gem: str, *, require_extracted: bool) -> list[str]:
    missing: list[str] = []
    base = DATA / gem
    if not base.is_dir():
        zip_path = DATA / f"{gem}.zip"
        hint = f" — unpack: cd Data && unzip -o {gem}.zip" if zip_path.is_file() else ""
        return [f"Data/{gem}/{hint}"]
    for pattern in REQUIRED_FILES:
        rel = pattern.format(gem=gem)
        if not (base / rel).is_file():
            missing.append(f"Data/{gem}/{rel}")
    if require_extracted:
        for sub in EXTRACTED_DIRS:
            d = base / sub
            if not d.is_dir() or not any(d.iterdir()):
                missing.append(f"Data/{gem}/{sub}/ (empty or missing)")
    return missing


def main() -> int:
    if not DATA.is_dir():
        print("MISSING  Data/", file=sys.stderr)
        print("See Data/README.md.", file=sys.stderr)
        return 1

    quick_start_only = "--quick-start-only" in sys.argv
    gems = (QUICK_START_GEM,) if quick_start_only else GEMS
    missing: list[str] = []
    for gem in gems:
        missing.extend(_check_gem(gem, require_extracted=True))

    if missing:
        for m in missing:
            print(f"MISSING  {m}")
        print(
            "\nUnpack datasets first:\n"
            "  cd Data && unzip -o e_coli_core.zip && unzip -o iMM904.zip && unzip -o iML1515.zip\n"
            "See Data/README.md.",
            file=sys.stderr,
        )
        return 1

    print("Data layout OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
