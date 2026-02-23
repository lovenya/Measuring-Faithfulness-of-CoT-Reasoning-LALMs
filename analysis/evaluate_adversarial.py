#!/usr/bin/env python3
"""
Analysis script for adversarial audio experiments.

Computes accuracy and consistency (with baseline) for each combination of:
- Track (animal, emotion, gender, language)
- Augmentation mode (concat, overlay) 
- Variant (correct, wrong)
- Hop type (single, multi, merged, or all)

Produces summary tables and optional CSV output.

Usage:
    python analysis/evaluate_adversarial.py --model qwen
    python analysis/evaluate_adversarial.py --model salmonn --hop-type single
    python analysis/evaluate_adversarial.py --model qwen --hop-type all
"""

import os
import json
import argparse
from collections import defaultdict


TRACKS = ['animal', 'emotion', 'gender', 'language']
AUG_MODES = ['concat', 'overlay']
VARIANTS = ['correct', 'wrong']


def load_results(filepath: str) -> list:
    """Load results from a JSONL file."""
    results = []
    if not os.path.exists(filepath):
        return results
    with open(filepath, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def get_hop_type(entry: dict) -> str | None:
    """Get hop_type from the entry, falling back to inferring from sample ID."""
    hop = entry.get('hop_type')
    if hop:
        return hop
    # Infer from ID: sakura_animal_0_single -> single, sakura_animal_0_multi -> multi
    sample_id = entry.get('id', '')
    if sample_id.endswith('_single'):
        return 'single'
    elif sample_id.endswith('_multi'):
        return 'multi'
    return None


def filter_by_hop_type(results: list, hop_type: str) -> list:
    """Filter results by hop_type. Returns all if hop_type is 'merged'."""
    if hop_type == 'merged':
        return results
    return [r for r in results if get_hop_type(r) == hop_type]


def compute_metrics(results: list) -> dict:
    """Compute accuracy and consistency metrics for a set of results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "accuracy": None, "consistency": None}
    
    correct = sum(1 for r in results if r.get('is_correct', False))
    
    # Consistency: only count entries that have baseline predictions
    consistency_entries = [r for r in results if r.get('corresponding_baseline_predicted_choice') is not None]
    consistent = sum(1 for r in consistency_entries if r.get('is_consistent_with_baseline', False))
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else None,
        "consistency_total": len(consistency_entries),
        "consistent": consistent,
        "consistency": consistent / len(consistency_entries) if len(consistency_entries) > 0 else None,
    }


def run_analysis(args, hop_type_label: str, hop_filter: str):
    """Run the full analysis for a given hop_type filter."""
    all_metrics = []
    
    print(f"\n{'=' * 80}")
    print(f"Adversarial Experiment Analysis — {args.model.upper()} — Hop: {hop_type_label.upper()}")
    print(f"{'=' * 80}")
    
    # Baseline accuracy for comparison
    print(f"\n{'─' * 80}")
    print(f"{'BASELINE ACCURACY (for comparison)':^80}")
    print(f"{'─' * 80}")
    print(f"{'Track':<12} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print(f"{'─' * 80}")
    
    baseline_acc = {}
    for track in TRACKS:
        baseline_path = os.path.join(args.baseline_dir, f"baseline_{args.model}_sakura-{track}.jsonl")
        baseline_results = filter_by_hop_type(load_results(baseline_path), hop_filter)
        if baseline_results:
            bc = sum(1 for r in baseline_results if r.get('is_correct', False))
            bt = len(baseline_results)
            baseline_acc[track] = bc / bt if bt > 0 else None
            print(f"{track:<12} {bt:>8} {bc:>8} {baseline_acc[track]:>10.2%}")
        else:
            baseline_acc[track] = None
            print(f"{track:<12} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
    
    # Main adversarial results
    for aug in AUG_MODES:
        print(f"\n{'═' * 80}")
        print(f"{'AUGMENTATION: ' + aug.upper():^80}")
        print(f"{'═' * 80}")
        print(f"{'Track':<12} {'Variant':<10} {'Total':>6} {'Acc':>8} {'Δ Acc':>8} {'Consist':>10} {'Consist %':>10}")
        print(f"{'─' * 80}")
        
        for variant in VARIANTS:
            for track in TRACKS:
                filename = f"adversarial_{args.model}_sakura-{track}_{aug}_{variant}.jsonl"
                filepath = os.path.join(args.results_dir, aug, filename)
                
                results = filter_by_hop_type(load_results(filepath), hop_filter)
                metrics = compute_metrics(results)
                
                if metrics['accuracy'] is not None:
                    acc_str = f"{metrics['accuracy']:.2%}"
                    if baseline_acc.get(track) is not None:
                        delta = metrics['accuracy'] - baseline_acc[track]
                        delta_str = f"{delta:+.2%}"
                    else:
                        delta_str = "N/A"
                else:
                    acc_str = "N/A"
                    delta_str = "N/A"
                
                if metrics['consistency'] is not None:
                    consist_str = f"{metrics['consistent']}/{metrics['consistency_total']}"
                    consist_pct = f"{metrics['consistency']:.2%}"
                else:
                    consist_str = "N/A"
                    consist_pct = "N/A"
                
                print(f"{track:<12} {variant:<10} {metrics['total']:>6} {acc_str:>8} {delta_str:>8} {consist_str:>10} {consist_pct:>10}")
                
                all_metrics.append({
                    'hop_type': hop_type_label,
                    'track': track,
                    'aug_mode': aug,
                    'variant': variant,
                    **metrics,
                    'baseline_accuracy': baseline_acc.get(track),
                })
    
    # Summary by augmentation mode
    print(f"\n{'═' * 80}")
    print(f"{'SUMMARY BY AUGMENTATION MODE':^80}")
    print(f"{'═' * 80}")
    print(f"{'Aug Mode':<12} {'Variant':<10} {'Avg Accuracy':>14} {'Avg Consistency':>16}")
    print(f"{'─' * 80}")
    
    for aug in AUG_MODES:
        for variant in VARIANTS:
            relevant = [m for m in all_metrics if m['aug_mode'] == aug and m['variant'] == variant and m['accuracy'] is not None]
            if relevant:
                avg_acc = sum(m['accuracy'] for m in relevant) / len(relevant)
                consist_vals = [m['consistency'] for m in relevant if m['consistency'] is not None]
                avg_consist = sum(consist_vals) / len(consist_vals) if consist_vals else None
                consist_str = f"{avg_consist:.2%}" if avg_consist is not None else "N/A"
                print(f"{aug:<12} {variant:<10} {avg_acc:>14.2%} {consist_str:>16}")
            else:
                print(f"{aug:<12} {variant:<10} {'N/A':>14} {'N/A':>16}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze adversarial experiment results.")
    parser.add_argument('--model', type=str, default='qwen',
                        help="Model alias (e.g., 'qwen', 'salmonn', 'salmonn_7b', 'flamingo').")
    parser.add_argument('--results-dir', type=str, default=None,
                        help="Base results directory (default: results/{model}/adversarial).")
    parser.add_argument('--baseline-dir', type=str, default=None,
                        help="Baseline results directory (default: results/{model}/baseline).")
    parser.add_argument('--hop-type', type=str, default='merged',
                        choices=['merged', 'single', 'multi', 'all'],
                        help="Hop type filter for Sakura datasets.\n"
                             "  merged = all data together (default, classic behavior)\n"
                             "  single = single-hop only\n"
                             "  multi  = multi-hop only\n"
                             "  all    = run both single and multi separately")
    parser.add_argument('--output-csv', type=str, default=None,
                        help="Optional CSV output file for results.")
    args = parser.parse_args()
    
    if args.results_dir is None:
        args.results_dir = f"results/{args.model}/adversarial"
    if args.baseline_dir is None:
        args.baseline_dir = f"results/{args.model}/baseline"
    
    # Determine which hop types to run
    if args.hop_type == 'all':
        hop_runs = [('single', 'single'), ('multi', 'multi')]
    else:
        hop_runs = [(args.hop_type, args.hop_type)]
    
    all_metrics = []
    for hop_label, hop_filter in hop_runs:
        metrics = run_analysis(args, hop_label, hop_filter)
        all_metrics.extend(metrics)
    
    # Optional CSV output
    if args.output_csv and all_metrics:
        import csv
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        fieldnames = ['hop_type', 'track', 'aug_mode', 'variant', 'total', 'correct', 'accuracy',
                     'consistency_total', 'consistent', 'consistency', 'baseline_accuracy']
        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)
        print(f"\nCSV output saved to: {args.output_csv}")
    
    print(f"\n{'=' * 80}")
    print("Analysis complete.")
    print("=" * 80)


if __name__ == '__main__':
    main()
