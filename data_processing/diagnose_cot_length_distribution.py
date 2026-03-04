# data_processing/diagnose_cot_length_distribution.py

"""
Diagnostic script to analyze the distribution of Chain-of-Thought (CoT) lengths
across models and datasets.

Outputs:
  - Per-dataset histogram of sentence counts
  - Summary statistics (mean, median, std, min, max)
  - Count / percentage of samples within a configurable range (default 1-7)
  - Optionally saves a bar chart PNG

Usage:
    python data_processing/diagnose_cot_length_distribution.py --model flamingo_hf --dataset all
    python data_processing/diagnose_cot_length_distribution.py --model flamingo_hf --dataset mmar --save-plot
"""

import os
import json
import argparse
import nltk
import numpy as np
from collections import Counter

# NLTK setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_nltk_data_path = os.path.join(PROJECT_ROOT, 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.insert(0, local_nltk_data_path)


def discover_datasets(model: str, results_dir: str) -> list[str]:
    """Find all datasets for a model by scanning the baseline directory."""
    baseline_dir = os.path.join(results_dir, model, 'baseline')
    datasets = set()
    for f in os.listdir(baseline_dir):
        if f.endswith('.jsonl') and '.part_' not in f and '-restricted' not in f:
            name = f.replace(f'baseline_{model}_', '').replace('.jsonl', '')
            datasets.add(name)
    return sorted(datasets)


def analyze_one_dataset(model: str, dataset: str, results_dir: str, 
                         num_chains: int, range_min: int, range_max: int) -> dict:
    """Analyze CoT sentence lengths for a single (model, dataset) pair."""
    
    baseline_path = os.path.join(results_dir, model, 'baseline', f'baseline_{model}_{dataset}.jsonl')
    if not os.path.exists(baseline_path):
        return None

    lengths = []
    with open(baseline_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            chain_id = data.get('chain_id', 0)
            if num_chains > 0 and chain_id >= num_chains:
                continue
            
            cot = data.get('sanitized_cot', '')
            n_sent = len(nltk.sent_tokenize(cot)) if cot and cot.strip() else 0
            lengths.append(n_sent)

    if not lengths:
        return None

    lengths_arr = np.array(lengths)
    counter = Counter(lengths)
    in_range = sum(1 for l in lengths if range_min <= l <= range_max)

    return {
        'dataset': dataset,
        'total': len(lengths),
        'mean': float(np.mean(lengths_arr)),
        'median': float(np.median(lengths_arr)),
        'std': float(np.std(lengths_arr)),
        'min': int(np.min(lengths_arr)),
        'max': int(np.max(lengths_arr)),
        'in_range': in_range,
        'in_range_pct': in_range / len(lengths) * 100,
        'out_of_range': len(lengths) - in_range,
        'out_of_range_pct': (len(lengths) - in_range) / len(lengths) * 100,
        'distribution': dict(sorted(counter.items())),
    }


def print_report(stats: dict, range_min: int, range_max: int):
    """Print a formatted report for one dataset."""
    ds = stats['dataset']
    
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds.upper()} ({stats['total']} samples)")
    print(f"{'='*60}")
    
    print(f"\n  Summary Statistics:")
    print(f"    Mean:   {stats['mean']:.2f} sentences")
    print(f"    Median: {stats['median']:.1f} sentences")
    print(f"    Std:    {stats['std']:.2f}")
    print(f"    Range:  {stats['min']} — {stats['max']} sentences")
    
    print(f"\n  Filtering ({range_min}-{range_max} sentences):")
    print(f"    IN range:  {stats['in_range']:5d} ({stats['in_range_pct']:5.1f}%)")
    print(f"    OUT range: {stats['out_of_range']:5d} ({stats['out_of_range_pct']:5.1f}%)")
    
    print(f"\n  Full Distribution:")
    print(f"    {'Length':>6}  {'Count':>6}  {'Pct':>6}  Bar")
    print(f"    {'------':>6}  {'------':>6}  {'------':>6}  ---")
    
    max_count = max(stats['distribution'].values()) if stats['distribution'] else 1
    for length in sorted(stats['distribution'].keys()):
        count = stats['distribution'][length]
        pct = count / stats['total'] * 100
        bar_width = int(count / max_count * 40)
        marker = " ◀ KEPT" if range_min <= length <= range_max else ""
        print(f"    {length:6d}  {count:6d}  {pct:5.1f}%  {'█' * bar_width}{marker}")


def save_plot(all_stats: list, model: str, plots_dir: str, range_min: int, range_max: int):
    """Save a bar chart of sentence length distributions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot.")
        return

    n_datasets = len(all_stats)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)
    fig.suptitle(f'CoT Sentence Length Distribution — {model.upper()}', fontsize=16, fontweight='bold')

    for idx, stats in enumerate(all_stats):
        ax = axes[0][idx]
        lengths = sorted(stats['distribution'].keys())
        counts = [stats['distribution'][l] for l in lengths]
        colors = ['#4CAF50' if range_min <= l <= range_max else '#F44336' for l in lengths]
        
        ax.bar(lengths, counts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(f"{stats['dataset'].upper()}\n({stats['in_range_pct']:.0f}% in {range_min}-{range_max})")
        ax.set_xlabel('Sentence Count')
        ax.set_ylabel('Number of Samples')
        ax.axvline(x=range_max + 0.5, color='red', linestyle='--', alpha=0.5, label=f'Max={range_max}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f'cot_length_distribution_{model}.png')
    plt.savefig(out_path, dpi=200)
    print(f"\n  Plot saved to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose CoT sentence length distribution across datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="Model alias, or 'all'.")
    parser.add_argument('--dataset', type=str, default='all', help="Dataset alias, or 'all'. Default: all.")
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--num-chains', type=int, default=1, help="Only analyze first N chains per question. Default: 1.")
    parser.add_argument('--min-sentences', type=int, default=1, help="Min sentence count for range analysis. Default: 1.")
    parser.add_argument('--max-sentences', type=int, default=7, help="Max sentence count for range analysis. Default: 7.")
    parser.add_argument('--save-plot', action='store_true', help="Save a bar chart PNG.")
    parser.add_argument('--plots-dir', type=str, default='plots/diagnostics', help="Directory for saved plots.")
    args = parser.parse_args()

    # Resolve models
    if args.model == 'all':
        models = sorted([
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d, 'baseline'))
        ])
    else:
        models = [args.model]

    for model in models:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model.upper()}")
        print(f"{'#'*60}")

        if args.dataset == 'all':
            datasets = discover_datasets(model, args.results_dir)
        else:
            datasets = [args.dataset]

        all_stats = []
        for dataset in datasets:
            stats = analyze_one_dataset(model, dataset, args.results_dir, 
                                         args.num_chains, args.min_sentences, args.max_sentences)
            if stats:
                all_stats.append(stats)
                print_report(stats, args.min_sentences, args.max_sentences)

        if all_stats:
            # Cross-dataset summary
            total_all = sum(s['total'] for s in all_stats)
            in_range_all = sum(s['in_range'] for s in all_stats)
            print(f"\n{'='*60}")
            print(f"  CROSS-DATASET SUMMARY for {model.upper()}")
            print(f"{'='*60}")
            print(f"  Total samples: {total_all}")
            print(f"  In range ({args.min_sentences}-{args.max_sentences}): {in_range_all} ({in_range_all/total_all*100:.1f}%)")
            print(f"  Out of range: {total_all - in_range_all} ({(total_all - in_range_all)/total_all*100:.1f}%)")
            
            # Per-dataset summary table
            print(f"\n  {'Dataset':<20} {'Total':>6} {'In Range':>10} {'Pct':>8} {'Mean':>6} {'Median':>8} {'Max':>5}")
            print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*5}")
            for s in all_stats:
                print(f"  {s['dataset']:<20} {s['total']:>6} {s['in_range']:>10} {s['in_range_pct']:>7.1f}% {s['mean']:>6.1f} {s['median']:>8.1f} {s['max']:>5}")

            if args.save_plot:
                save_plot(all_stats, model, args.plots_dir, args.min_sentences, args.max_sentences)


if __name__ == "__main__":
    main()
