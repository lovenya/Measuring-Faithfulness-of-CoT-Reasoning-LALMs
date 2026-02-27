#!/usr/bin/env python3
# experiments/testing/analyze_variance.py

"""
Variance Analysis Script

Computes variance metrics for baseline results and optionally compares them
against prompt variance test results. Designed to help diagnose why models
show high variance across chains.

Metrics computed per results file:
  - Overall accuracy
  - Per-sample accuracy mean/std
  - Mixed samples % (samples where chains disagree)
  - Avg unique choices per sample
  - Per-sample entropy (information-theoretic spread)
  - Fleiss' Kappa (inter-chain agreement)

Usage:
    # Baseline only:
    python experiments/testing/analyze_variance.py --model qwen --dataset mmar

    # Compare baseline vs a specific test run:
    python experiments/testing/analyze_variance.py --model qwen --dataset mmar \\
        --strategy two_turn_cot --temperature 0.1

    # Compare baseline vs all test runs found in test/ folder:
    python experiments/testing/analyze_variance.py --model qwen --dataset mmar --all

    # With explicit paths:
    python experiments/testing/analyze_variance.py \\
        --baseline-path results/qwen/baseline/baseline_qwen_mmar.jsonl \\
        --test-path results/qwen/test/variance_no_cot_t0.1_qwen_mmar.jsonl

    # Save summary to file:
    python experiments/testing/analyze_variance.py --model qwen --dataset mmar --output summary.txt
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_results(filepath: str) -> list[dict]:
    """Load JSONL results file."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def compute_entropy(counts: list[int]) -> float:
    """Shannon entropy of a discrete distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def compute_fleiss_kappa(choice_matrix: list[list[int]], num_categories: int) -> float:
    """
    Compute Fleiss' Kappa for inter-chain agreement.
    choice_matrix: list of lists, where each inner list contains the count
                   of each category chosen by raters (chains) for that subject (sample).
    """
    n_subjects = len(choice_matrix)
    if n_subjects == 0:
        return 0.0
    n_raters = sum(choice_matrix[0])
    if n_raters <= 1:
        return 1.0  # perfect agreement with 1 rater

    # P_i: proportion of agreement for each subject
    P_i = []
    for row in choice_matrix:
        n = sum(row)
        if n <= 1:
            P_i.append(1.0)
        else:
            P_i.append((sum(c * c for c in row) - n) / (n * (n - 1)))

    P_bar = np.mean(P_i)

    # P_e: expected agreement by chance
    category_proportions = []
    total_ratings = n_subjects * n_raters
    for j in range(num_categories):
        p_j = sum(row[j] for row in choice_matrix) / total_ratings
        category_proportions.append(p_j)
    P_e = sum(p * p for p in category_proportions)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def compute_metrics(entries: list[dict], label: str = "") -> dict:
    """Compute all variance metrics for a set of results."""
    if not entries:
        return {"label": label, "error": "No entries found"}

    # Group by sample ID
    by_id = defaultdict(list)
    for e in entries:
        by_id[e['id']].append(e)

    total_samples = len(by_id)
    chains_counts = [len(v) for v in by_id.values()]

    # Overall accuracy
    all_correct = [e.get('is_correct', False) for e in entries]
    overall_acc = np.mean(all_correct) * 100

    # Per-sample accuracy and intra-sample variance
    per_sample_acc = []
    per_sample_var = []
    for sid, trials in by_id.items():
        correct = [int(t.get('is_correct', False)) for t in trials]
        per_sample_acc.append(np.mean(correct) * 100)
        per_sample_var.append(np.var(correct))

    # Mixed samples
    mixed = 0
    always_correct = 0
    always_wrong = 0
    for sid, trials in by_id.items():
        correct_set = set(t.get('is_correct', False) for t in trials)
        if len(correct_set) > 1:
            mixed += 1
        elif True in correct_set:
            always_correct += 1
        else:
            always_wrong += 1

    # Choice diversity
    choices_by_id = defaultdict(list)
    for e in entries:
        choices_by_id[e['id']].append(e.get('predicted_choice', '?'))
    unique_choices = [len(set(v)) for v in choices_by_id.values()]

    # # Per-sample entropy
    # entropies = []
    # for sid, preds in choices_by_id.items():
    #     counts = list(Counter(preds).values())
    #     entropies.append(compute_entropy(counts))

    # # Fleiss' Kappa
    # all_possible_choices = sorted(set(
    #     (e.get('predicted_choice') or '?') for e in entries
    # ))
    # choice_to_idx = {c: i for i, c in enumerate(all_possible_choices)}
    # num_categories = len(all_possible_choices)

    # choice_matrix = []
    # for sid in by_id:
    #     preds = choices_by_id[sid]
    #     row = [0] * num_categories
    #     for p in preds:
    #         row[choice_to_idx.get(p, 0)] += 1
    #     choice_matrix.append(row)

    # kappa = compute_fleiss_kappa(choice_matrix, num_categories)

    return {
        "label": label,
        "total_entries": len(entries),
        "total_samples": total_samples,
        "chains_per_sample": f"{min(chains_counts)}-{max(chains_counts)}",
        "overall_accuracy": overall_acc,
        "per_sample_acc_mean": np.mean(per_sample_acc),
        "mean_intra_sample_variance_pct": np.mean(per_sample_var) * 4 * 100,  # 0.25 (max variance) * 4 * 100 = 100%
        "always_correct": always_correct,
        "always_correct_pct": always_correct / total_samples * 100,
        "always_wrong": always_wrong,
        "always_wrong_pct": always_wrong / total_samples * 100,
        "mixed": mixed,
        "mixed_pct": mixed / total_samples * 100,
        "avg_unique_choices": np.mean(unique_choices),
        # "avg_entropy": np.mean(entropies),
        # "fleiss_kappa": kappa,
    }


def format_metrics(m: dict) -> str:
    """Format metrics dict into a readable text block."""
    if "error" in m:
        return f"\n  {m['label']}: {m['error']}\n"

    lines = []
    lines.append(f"\n  {'─'*55}")
    lines.append(f"  {m['label']}")
    lines.append(f"  {'─'*55}")
    lines.append(f"  Entries:              {m['total_entries']} ({m['total_samples']} samples × {m['chains_per_sample']} chains)")
    lines.append(f"  Overall Accuracy:     {m['overall_accuracy']:.1f}%")
    lines.append(f"  Per-Sample Acc Mean:  {m['per_sample_acc_mean']:.1f}%")
    lines.append(f"  Mean Intra-Variance:  {m['mean_intra_sample_variance_pct']:.1f}%   ← key metric (0%=stable, 100%=random)")
    lines.append(f"  Always Correct:       {m['always_correct']:>4} ({m['always_correct_pct']:.1f}%)")
    lines.append(f"  Always Wrong:         {m['always_wrong']:>4} ({m['always_wrong_pct']:.1f}%)")
    lines.append(f"  Mixed (inconsistent): {m['mixed']:>4} ({m['mixed_pct']:.1f}%)  ← key metric")
    lines.append(f"  Avg Unique Choices:   {m['avg_unique_choices']:.2f}")
    # lines.append(f"  Avg Entropy:          {m['avg_entropy']:.3f} bits")
    # lines.append(f"  Fleiss' Kappa:        {m['fleiss_kappa']:.3f}  (1.0=perfect, 0=chance)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze variance in baseline and prompt variance test results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model', type=str, default=None,
                        help="Model alias (e.g., 'qwen'). Used to auto-build paths.")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Dataset name (e.g., 'mmar'). Used to auto-build paths.")
    parser.add_argument('--baseline-path', type=str, default=None,
                        help="Explicit path to baseline results JSONL.")
    parser.add_argument('--test-path', type=str, default=None,
                        help="Explicit path to test results JSONL.")
    parser.add_argument('--strategy', type=str, default=None,
                        help="Strategy name to auto-build test path (e.g., 'no_cot').")
    parser.add_argument('--temperature', type=float, default=None,
                        help="Temperature to auto-build test path (e.g., 0.1).")
    parser.add_argument('--top-p', type=float, default=None,
                        help="Top-p to auto-build test path (e.g., 0.9).")
    parser.add_argument('--top-k', type=int, default=None,
                        help="Top-k to auto-build test path (e.g., 50).")
    parser.add_argument('--all', action='store_true',
                        help="Compare baseline against ALL test results found in test/ folder.")
    parser.add_argument('--results-dir', type=str, default='results',
                        help="Base results directory (default: results).")
    parser.add_argument('--output', type=str, default=None,
                        help="Optional path to save text summary.")
    args = parser.parse_args()

    output_lines = []

    def log(text):
        print(text)
        output_lines.append(text)

    log("=" * 60)
    log("VARIANCE ANALYSIS")
    log("=" * 60)

    # --- Load baseline ---
    baseline_path = args.baseline_path
    if not baseline_path and args.model and args.dataset:
        baseline_path = os.path.join(
            args.results_dir, args.model, 'baseline',
            f'baseline_{args.model}_{args.dataset}.jsonl'
        )

    if baseline_path and os.path.exists(baseline_path):
        log(f"\nBaseline: {baseline_path}")
        baseline_entries = load_results(baseline_path)
        baseline_metrics = compute_metrics(baseline_entries, f"BASELINE (temp=1.0, two_turn_cot)")
        log(format_metrics(baseline_metrics))
    else:
        log(f"\nWARNING: Baseline not found: {baseline_path}")
        baseline_metrics = None

    # --- Load test results ---
    test_files = []

    if args.test_path:
        test_files.append(args.test_path)
    elif args.all and args.model and args.dataset:
        # Find all matching test files
        test_dir = os.path.join(args.results_dir, args.model, 'test')
        pattern = os.path.join(test_dir, f'variance_*_{args.model}_{args.dataset}.jsonl')
        test_files = sorted(glob.glob(pattern))
        if not test_files:
            log(f"\nNo test results found matching: {pattern}")
    elif args.strategy and args.model and args.dataset:
        temp = args.temperature
        top_p = args.top_p
        top_k = args.top_k
        
        # Apply defaults based on model
        if args.model == 'qwen':
            if temp is None: temp = 1.0
            if top_p is None: top_p = 0.01
            if top_k is None: top_k = 0
        elif args.model in ('salmonn', 'salmonn_7b'):
            if temp is None: temp = 1.0
            if top_p is None: top_p = 0.9
            if top_k is None: top_k = 50
        elif args.model in ('flamingo', 'flamingo_hf'):
            if temp is None: temp = 0.7
            if top_p is None: top_p = 0.8
            if top_k is None: top_k = 20
        else:
            if temp is None: temp = 1.0
            if top_p is None: top_p = 0.9
            if top_k is None: top_k = 50

        test_path = os.path.join(
            args.results_dir, args.model, 'test',
            f'variance_{args.strategy}_t{temp}_p{top_p}_k{top_k}_{args.model}_{args.dataset}.jsonl'
        )
        test_files.append(test_path)

    for tf in test_files:
        if os.path.exists(tf):
            log(f"\nTest: {tf}")
            test_entries = load_results(tf)
            # Extract label from filename
            fname = Path(tf).stem  # e.g., variance_no_cot_t0.1_qwen_mmar
            test_metrics = compute_metrics(test_entries, fname.upper())
            log(format_metrics(test_metrics))
        else:
            log(f"\nWARNING: Test file not found: {tf}")

    # --- Summary comparison ---
    if baseline_metrics and test_files:
        log(f"\n{'=' * 60}")
        log("COMPARISON SUMMARY")
        log(f"{'=' * 60}")
        log(f"  {'Metric':<25} {'Baseline':>10}")
        log(f"  {'-'*55}")
        log(f"  {'Accuracy':<25} {baseline_metrics['overall_accuracy']:>9.1f}%")
        log(f"  {'Mean Intra-Variance':<25} {baseline_metrics['mean_intra_sample_variance_pct']:>9.1f}%")
        log(f"  {'Mixed %':<25} {baseline_metrics['mixed_pct']:>9.1f}%")
        log(f"  {'Avg Unique Choices':<25} {baseline_metrics['avg_unique_choices']:>10.2f}")
        # log(f"  {'Fleiss Kappa':<25} {baseline_metrics['fleiss_kappa']:>10.3f}")

    log(f"\n{'=' * 60}")

    # Save if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"\n✓ Summary saved to: {args.output}")


if __name__ == '__main__':
    main()
