#!/usr/bin/env python3
"""
Cross-dataset plotting script for adversarial audio experiments.

Generates plots comparing all 4 tracks on the same graph:
- One plot per (aug_mode, variant) combination
- Each track is a separate bar
- Shows both accuracy and consistency side-by-side
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRACK_STYLES = {
    "animal":   {"label": "S.Animal",   "color": "#377eb8"},
    "emotion":  {"label": "S.Emotion",  "color": "#4daf4a"},
    "gender":   {"label": "S.Gender",   "color": "#ff7f00"},
    "language": {"label": "S.Language", "color": "#984ea3"},
}


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


def compute_metrics(results: list) -> dict:
    """Compute accuracy and consistency metrics."""
    total = len(results)
    if total == 0:
        return {"total": 0, "accuracy": None, "consistency": None}
    
    correct = sum(1 for r in results if r.get('is_correct', False))
    consistency_entries = [r for r in results if r.get('corresponding_baseline_predicted_choice') is not None]
    consistent = sum(1 for r in consistency_entries if r.get('is_consistent_with_baseline', False))
    
    return {
        "total": total,
        "accuracy": correct / total if total > 0 else None,
        "consistency": consistent / len(consistency_entries) if len(consistency_entries) > 0 else None,
    }


def plot_cross_dataset(
    aug_mode: str,
    variant: str,
    model_name: str,
    results_dir: str,
    baseline_dir: str,
    plots_dir: str,
    save_pdf: bool = False,
):
    """Generate a cross-dataset plot for one (aug_mode, variant) combination."""
    
    print(f"\n--- Cross-Dataset Plot: {aug_mode.upper()} / {variant.upper()} ---")
    
    tracks = list(TRACK_STYLES.keys())
    accuracies = []
    consistencies = []
    baseline_accs = []
    labels = []
    colors = []
    
    for track in tracks:
        # Load adversarial results
        filename = f"adversarial_{model_name}_sakura-{track}_{aug_mode}_{variant}.jsonl"
        filepath = os.path.join(results_dir, aug_mode, filename)
        results = load_results(filepath)
        metrics = compute_metrics(results)
        
        # Load baseline
        baseline_path = os.path.join(baseline_dir, f"baseline_{model_name}_sakura-{track}.jsonl")
        baseline_results = load_results(baseline_path)
        baseline_metrics = compute_metrics(baseline_results)
        
        if metrics['accuracy'] is not None:
            accuracies.append(metrics['accuracy'] * 100)
            consistencies.append(metrics['consistency'] * 100 if metrics['consistency'] is not None else 0)
            baseline_accs.append(baseline_metrics['accuracy'] * 100 if baseline_metrics['accuracy'] is not None else 0)
            labels.append(TRACK_STYLES[track]['label'])
            colors.append(TRACK_STYLES[track]['color'])
            print(f"  {track}: Acc={metrics['accuracy']*100:.1f}%, Consist={metrics['consistency']*100:.1f}%")
        else:
            print(f"  {track}: No data")
    
    if not accuracies:
        print(f"  No data found. Skipping plot.")
        return
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Accuracy plot with baseline comparison
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Adversarial', color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars_base = ax1.bar(x + width/2, baseline_accs, width, label='Baseline', color='gray', alpha=0.6, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Adversarial Audio Accuracy: {aug_mode.capitalize()} + {variant.capitalize()}\n({model_name.upper()})', 
                  fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars_base:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Consistency plot
    bars2 = ax2.bar(x, consistencies, width*2, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Consistency with Baseline (%)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Consistency with Baseline: {aug_mode.capitalize()} + {variant.capitalize()}\n({model_name.upper()})', 
                  fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(plots_dir, model_name, 'adversarial')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"cross_dataset_adversarial_{model_name}_{aug_mode}_{variant}"
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {png_path}")
    
    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"  ✓ Saved: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate cross-dataset adversarial plots.")
    parser.add_argument('--model', type=str, default='qwen', help="Model name")
    parser.add_argument('--aug-mode', type=str, default='all', choices=['concat', 'overlay', 'all'],
                        help="Augmentation mode to plot")
    parser.add_argument('--variant', type=str, default='all', choices=['correct', 'wrong', 'all'],
                        help="Variant to plot")
    parser.add_argument('--results-dir', type=str, default='results/qwen/adversarial',
                        help="Adversarial results directory")
    parser.add_argument('--baseline-dir', type=str, default=None,
                        help="Baseline results directory")
    parser.add_argument('--plots-dir', type=str, default='plots/cross_dataset_plots',
                        help="Output directory")
    parser.add_argument('--save-pdf', action='store_true', help="Save PDF copies")
    
    args = parser.parse_args()
    
    if args.baseline_dir is None:
        args.baseline_dir = f"results/{args.model}/baseline"
    
    aug_modes = ['concat', 'overlay'] if args.aug_mode == 'all' else [args.aug_mode]
    variants = ['correct', 'wrong'] if args.variant == 'all' else [args.variant]
    
    print("=" * 60)
    print(f"Cross-Dataset Adversarial Plotting ({args.model.upper()})")
    print("=" * 60)
    
    for aug in aug_modes:
        for var in variants:
            plot_cross_dataset(
                aug, var, args.model, args.results_dir, args.baseline_dir,
                args.plots_dir, args.save_pdf
            )
    
    print("\n" + "=" * 60)
    print("Plotting complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
