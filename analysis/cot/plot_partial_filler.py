#!/usr/bin/env python3
"""Cross-dataset plotting for partial_filler_text with mode selector."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.cot.plot_cot_experiment import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial Filler cross-dataset analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="start", choices=["start", "end", "random"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--reference-mode", default="0pct", choices=["0pct", "baseline", "100pct"])
    parser.add_argument("--filler-type", default="dots", choices=["dots", "lorem"])
    parser.add_argument("--bin-size", type=int, default=5)
    parser.add_argument("--min-bin-count", type=int, default=1)
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument("--show-ci", action="store_true")
    parser.add_argument("--print-line-data", action="store_true")
    parser.add_argument("--save-stats", action="store_true")
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--y-zoom", nargs=2, type=float, default=None)

    args = parser.parse_args()
    args.experiment = "partial_filler_text"
    args.partial_filler_mode = args.mode
    args.perturbation_source = "self"
    args.include_100_anchor = False
    run_analysis(args)


if __name__ == "__main__":
    main()
