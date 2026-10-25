#!/usr/bin/env python3
"""
Quick run script for the GraphGdel baseline on the e_coli_core model (default).

Delegates to baseline.GraphGdel_main.main with default --CBM e_coli_core and --use_cpu 1;
weights load from baseline/GraphGdel_<CBM>.sav when present, else random init. Additional CLI
arguments are forwarded (e.g. --split_seed, --output_metrics_csv, --train 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseline.GraphGdel_main import main as graphgdel_main


def run() -> None:
    prog = sys.argv[0] if sys.argv else "quick_run_baseline_GraphGdel.py"
    if len(sys.argv) <= 1:
        sys.argv = [prog, "--CBM", "e_coli_core", "--use_cpu", "1"]
    else:
        sys.argv = [prog, "--CBM", "e_coli_core", "--use_cpu", "1"] + sys.argv[1:]
    graphgdel_main()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
