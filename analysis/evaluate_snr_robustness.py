#!/usr/bin/env python3
"""
SNR Robustness Evaluation Script

Analyzes results from the SNR robustness experiment, computing accuracy
and consistency metrics at each SNR level. Optionally saves to CSV.

Usage:
    python analysis/evaluate_snr_robustness.py --model qwen
    python analysis/evaluate_snr_robustness.py --model qwen --datasets mmar sakura-animal
    python analysis/evaluate_snr_robustness.py --model qwen --output-csv results/qwen/snr_summary.csv
"""

import os
import sys
import json
import argparse
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SNR_LEVELS = ['clean', 20, 10, 5, 0, -5, -10]

ALL_DATASETS = ['mmar', 'sakura-animal', 'sakura-emotion', 'sakura-gender', 'sakura-language']


def load_results(results_dir: str, model: str, dataset: str) -> list[dict]:
    """Load SNR robustness results for a model/dataset pair."""
    filepath = os.path.join(
        results_dir, model, 'snr_robustness',
        f'snr_robustness_{model}_{dataset}.jsonl'
    )
    if not os.path.exists(filepath):
        return []
    
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def compute_metrics(entries: list[dict]) -> dict:
    """
    Compute accuracy and consistency per SNR level.
    Returns dict: snr_level -> {total, correct, consistent, accuracy, consistency}
    """
    by_snr = defaultdict(lambda: {'total': 0, 'correct': 0, 'consistent': 0})
    
    for entry in entries:
        snr = entry['snr_db']
        by_snr[snr]['total'] += 1
        if entry.get('is_correct'):
            by_snr[snr]['correct'] += 1
        if entry.get('is_consistent_with_baseline'):
            by_snr[snr]['consistent'] += 1
    
    metrics = {}
    for snr in SNR_LEVELS:
        data = by_snr.get(snr, {'total': 0, 'correct': 0, 'consistent': 0})
        total = data['total']
        if total > 0:
            metrics[snr] = {
                'total': total,
                'correct': data['correct'],
                'consistent': data['consistent'],
                'accuracy': data['correct'] / total * 100,
                'consistency': data['consistent'] / total * 100,
            }
        else:
            metrics[snr] = None
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SNR robustness experiment results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Model alias (e.g., 'qwen', 'salmonn').")
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        help="Datasets to evaluate (default: all).")
    parser.add_argument('--results-dir', type=str, default='results',
                        help="Base results directory.")
    parser.add_argument('--output-csv', type=str, default=None,
                        help="Optional: save summary to CSV.")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"SNR Robustness Analysis — {args.model.upper()}")
    print(f"{'='*70}")

    all_rows = []  # For CSV output

    for dataset in args.datasets:
        entries = load_results(args.results_dir, args.model, dataset)
        if not entries:
            print(f"\n--- {dataset.upper()} ---")
            print(f"  No results found.")
            continue

        metrics = compute_metrics(entries)

        print(f"\n--- {dataset.upper()} ({len(entries)} entries) ---")
        print(f"  {'SNR':>8} {'Total':>7} {'Acc':>8} {'Consist':>10} {'Consist %':>10}")
        print(f"  {'-'*50}")

        for snr in SNR_LEVELS:
            m = metrics.get(snr)
            if m:
                snr_label = 'clean' if snr == 'clean' else f'{snr}dB'
                print(f"  {snr_label:>8} {m['total']:>7} {m['accuracy']:>7.1f}% "
                      f"{m['consistent']}/{m['total']:>5} {m['consistency']:>9.1f}%")
                
                all_rows.append({
                    'model': args.model,
                    'dataset': dataset,
                    'snr_db': snr,
                    'total': m['total'],
                    'correct': m['correct'],
                    'accuracy': round(m['accuracy'], 2),
                    'consistent': m['consistent'],
                    'consistency': round(m['consistency'], 2),
                })

    # Save CSV if requested
    if args.output_csv and all_rows:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n✓ Summary saved to: {args.output_csv}")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
