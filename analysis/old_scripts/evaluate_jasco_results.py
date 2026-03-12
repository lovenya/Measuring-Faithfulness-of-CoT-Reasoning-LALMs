#!/usr/bin/env python3
"""
JASCO Results Analysis Script

Reads the LLM-as-a-Judge evaluation output and produces:
  1. Console text summary — per-condition avg score and 0/1/2 breakdown
  2. Line plot     — avg judge score vs masking % (audio vs speech masking)
  3. Stacked bar   — score distribution (0/1/2) per condition

Conditions:
  baseline       = unmasked audio (anchor)
  audio_10..100  = audio masked N% (speech track preserved)
  speech_10..100 = speech masked N% (audio track preserved)

Usage:
    python analysis/evaluate_jasco_results.py --model qwen
    python analysis/evaluate_jasco_results.py --model qwen --judge mistral --save-pdf
    python analysis/evaluate_jasco_results.py --model qwen --output-txt results/qwen/jasco_masking/summary.txt
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- Style ---
MASK_PERCENTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

SCORE_COLORS = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}  # red, orange, green
SCORE_LABELS = {0: "Score 0 (wrong)", 1: "Score 1 (partial)", 2: "Score 2 (correct)"}

LINE_STYLES = {
    "audio":   {"color": "#1f77b4", "linestyle": "-",  "marker": "o", "label": "Audio masked"},
    "speech":  {"color": "#d62728", "linestyle": "--", "marker": "s", "label": "Speech masked"},
    "baseline":{"color": "#2ca02c", "linestyle": ":",  "marker": "*", "label": "Baseline (no mask)"},
}


def load_results(input_path: str) -> list[dict]:
    entries = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def parse_condition(condition: str) -> tuple[str, int]:
    """
    Returns (mask_type, mask_percent).
    baseline -> ('baseline', 0)
    audio_50 -> ('audio', 50)
    speech_30 -> ('speech', 30)
    """
    if condition == 'baseline':
        return ('baseline', 0)
    parts = condition.rsplit('_', 1)
    return (parts[0], int(parts[1]))


def aggregate(entries: list[dict]) -> dict:
    """
    Aggregate scores by condition.
    Returns: {condition -> {'scores': [...], 'avg': float, 'dist': {0,1,2: count}}}
    """
    by_cond = defaultdict(list)
    for e in entries:
        score = e.get('judge_score')
        if score is not None:
            by_cond[e['condition']].append(score)

    result = {}
    for cond, scores in by_cond.items():
        n = len(scores)
        avg = np.mean(scores)
        dist = {0: scores.count(0), 1: scores.count(1), 2: scores.count(2)}
        result[cond] = {'n': n, 'avg': avg, 'dist': dist, 'scores': scores}
    return result


def print_summary(agg: dict, model: str, judge: str, output_txt: str | None = None):
    lines = []
    lines.append("=" * 70)
    lines.append(f"JASCO Evaluation Summary — {model.upper()} (judge: {judge})")
    lines.append("=" * 70)

    # Baseline
    baseline = agg.get('baseline')
    if baseline:
        lines.append(f"\n{'BASELINE (no masking)':}")
        lines.append(f"  N={baseline['n']}  Avg score: {baseline['avg']:.3f}/2.0")
        d = baseline['dist']
        lines.append(f"  Score 0: {d[0]:3d} ({d[0]/baseline['n']*100:.1f}%)  "
                     f"Score 1: {d[1]:3d} ({d[1]/baseline['n']*100:.1f}%)  "
                     f"Score 2: {d[2]:3d} ({d[2]/baseline['n']*100:.1f}%)")

    for mask_type in ['audio', 'speech']:
        lines.append(f"\n{'─'*70}")
        lines.append(f"{'MASK TYPE: ' + mask_type.upper()}")
        lines.append(f"{'─'*70}")
        lines.append(f"  {'%Masked':>8}  {'N':>5}  {'AvgScore':>9}  {'Score0':>7}  {'Score1':>7}  {'Score2':>7}")
        lines.append(f"  {'-'*62}")

        for pct in MASK_PERCENTS:
            if pct == 0:
                continue  # that's baseline
            cond = f"{mask_type}_{pct}"
            if cond not in agg:
                lines.append(f"  {pct:>7}%  {'N/A':>5}")
                continue
            a = agg[cond]
            d = a['dist']
            lines.append(
                f"  {pct:>7}%  {a['n']:>5}  {a['avg']:>9.3f}  "
                f"{d[0]:>5} ({d[0]/a['n']*100:.0f}%)  "
                f"{d[1]:>5} ({d[1]/a['n']*100:.0f}%)  "
                f"{d[2]:>5} ({d[2]/a['n']*100:.0f}%)"
            )

    lines.append(f"\n{'='*70}")

    text = "\n".join(lines)
    print(text)

    if output_txt:
        os.makedirs(os.path.dirname(output_txt) or '.', exist_ok=True)
        with open(output_txt, 'w') as f:
            f.write(text + "\n")
        print(f"\n✓ Text summary saved to: {output_txt}")


def plot_avg_score(agg: dict, model: str, judge: str, plots_dir: str, save_pdf: bool):
    """Line plot: avg score vs masking % for audio and speech conditions."""
    fontsize = 18
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Baseline horizontal reference
    baseline_avg = agg.get('baseline', {}).get('avg', None)
    if baseline_avg is not None:
        ax.axhline(baseline_avg,
                   color=LINE_STYLES['baseline']['color'],
                   linestyle=LINE_STYLES['baseline']['linestyle'],
                   linewidth=1.5, label=f"Baseline ({baseline_avg:.2f})")

    for mask_type in ['audio', 'speech']:
        xs, ys = [], []
        for pct in MASK_PERCENTS:
            if pct == 0:
                continue
            cond = f"{mask_type}_{pct}"
            if cond in agg:
                xs.append(pct)
                ys.append(agg[cond]['avg'])
        style = LINE_STYLES[mask_type]
        ax.plot(xs, ys, label=style['label'], color=style['color'],
                linestyle=style['linestyle'], marker=style['marker'],
                linewidth=2, markersize=8)

    ax.set_title(f"JASCO: Avg Judge Score vs Masking %\n{model.upper()} (judge: {judge})", fontsize=fontsize)
    ax.set_xlabel("% of Audio Masked", fontsize=fontsize)
    ax.set_ylabel("Avg Judge Score (0–2)", fontsize=fontsize)
    ax.set_xlim(5, 105)
    ax.set_ylim(-0.05, 2.15)
    ax.set_xticks(MASK_PERCENTS[1:])
    ax.tick_params(labelsize=fontsize - 4)
    ax.legend(fontsize=fontsize - 4)
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    base = os.path.join(plots_dir, f"jasco_avg_score_{model}_{judge}")
    plt.savefig(f"{base}.png", dpi=300)
    print(f"✓ Line plot saved: {base}.png")
    if save_pdf:
        plt.savefig(f"{base}.pdf", format='pdf')
        print(f"✓ PDF saved: {base}.pdf")
    plt.close()


def plot_score_distribution(agg: dict, model: str, judge: str, plots_dir: str, save_pdf: bool):
    """Stacked bar: 0/1/2 distribution per condition (ordered by mask type + %)."""
    # Build ordered condition list
    conditions = ['baseline']
    labels = ['Baseline']
    for mask_type in ['audio', 'speech']:
        for pct in MASK_PERCENTS[1:]:
            cond = f"{mask_type}_{pct}"
            if cond in agg:
                conditions.append(cond)
                labels.append(f"{mask_type[0].upper()}{pct}%")

    n0 = [agg[c]['dist'][0] / agg[c]['n'] * 100 for c in conditions if c in agg]
    n1 = [agg[c]['dist'][1] / agg[c]['n'] * 100 for c in conditions if c in agg]
    n2 = [agg[c]['dist'][2] / agg[c]['n'] * 100 for c in conditions if c in agg]
    labels = [l for c, l in zip(conditions, labels) if c in agg]

    fontsize = 13
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    x = np.arange(len(labels))
    width = 0.65

    bars0 = ax.bar(x, n0, width, label=SCORE_LABELS[0], color=SCORE_COLORS[0])
    bars1 = ax.bar(x, n1, width, bottom=n0, label=SCORE_LABELS[1], color=SCORE_COLORS[1])
    bars2 = ax.bar(x, n2, width, bottom=[a + b for a, b in zip(n0, n1)],
                   label=SCORE_LABELS[2], color=SCORE_COLORS[2])

    # Separator line between audio and speech groups
    audio_count = len([c for c in conditions if c != 'baseline' and 'audio' in c])
    ax.axvline(audio_count + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(audio_count + 0.6, 105, 'speech →', fontsize=10, alpha=0.6)

    ax.set_title(f"JASCO: Score Distribution by Condition\n{model.upper()} (judge: {judge})", fontsize=fontsize + 2)
    ax.set_xlabel("Condition", fontsize=fontsize)
    ax.set_ylabel("% of Responses", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=fontsize - 2)
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    base = os.path.join(plots_dir, f"jasco_score_dist_{model}_{judge}")
    plt.savefig(f"{base}.png", dpi=300)
    print(f"✓ Distribution plot saved: {base}.png")
    if save_pdf:
        plt.savefig(f"{base}.pdf", format='pdf')
        print(f"✓ PDF saved: {base}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JASCO LLM-judge evaluation results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Model alias (e.g., 'qwen', 'salmonn', 'flamingo').")
    parser.add_argument('--judge', type=str, default='mistral',
                        help="Judge name used during evaluation (default: mistral).")
    parser.add_argument('--results-dir', type=str, default='results',
                        help="Base results directory (default: results).")
    parser.add_argument('--plots-dir', type=str, default=None,
                        help="Output directory for plots (default: plots/{model}/jasco).")
    parser.add_argument('--output-txt', type=str, default=None,
                        help="Optional path to save text summary.")
    parser.add_argument('--save-pdf', action='store_true',
                        help="Also save plots as PDF.")
    parser.add_argument('--no-plots', action='store_true',
                        help="Skip plot generation (text summary only).")
    args = parser.parse_args()

    # Build input path
    input_path = os.path.join(
        args.results_dir, args.model, 'jasco_masking', 'llm_judge_evaluations',
        f'jasco_masking_{args.model}_jasco_evaluated_by_{args.judge}.jsonl'
    )
    if not os.path.exists(input_path):
        print(f"ERROR: Evaluated results not found: {input_path}")
        print("Run analysis/evaluate_jasco.py first.")
        sys.exit(1)

    if args.plots_dir is None:
        args.plots_dir = os.path.join('plots', args.model, 'jasco')

    print(f"Loading: {input_path}")
    entries = load_results(input_path)
    print(f"Loaded {len(entries)} entries (null scores: {sum(1 for e in entries if e.get('judge_score') is None)})")

    agg = aggregate(entries)

    print_summary(agg, args.model, args.judge, args.output_txt)

    if not args.no_plots:
        plot_avg_score(agg, args.model, args.judge, args.plots_dir, args.save_pdf)
        plot_score_distribution(agg, args.model, args.judge, args.plots_dir, args.save_pdf)


if __name__ == '__main__':
    main()
