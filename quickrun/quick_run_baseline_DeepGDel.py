#!/usr/bin/env python3
"""
Quick run script for the DeepGDel baseline on the e_coli_core model (default).

Delegates to baseline.DeepGDel_main.main with default --CBM e_coli_core and --use_cpu 1;
loads baseline/DeepGdel_<CBM>.sav when present, else random init. Additional CLI arguments are forwarded.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseline.DeepGDel_main import main as deepgdel_main


def run() -> None:
    prog = sys.argv[0] if sys.argv else "quick_run_baseline_DeepGDel.py"
    if len(sys.argv) <= 1:
        sys.argv = [prog, "--CBM", "e_coli_core", "--use_cpu", "1"]
    else:
        sys.argv = [prog, "--CBM", "e_coli_core", "--use_cpu", "1"] + sys.argv[1:]
    deepgdel_main()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
