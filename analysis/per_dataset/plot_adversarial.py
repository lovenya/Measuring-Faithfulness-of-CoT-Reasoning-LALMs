#!/usr/bin/env python3
"""
Per-dataset plotting script for adversarial audio experiments.

Generates plots showing:
- Accuracy for each augmentation mode (concat vs overlay) and variant (correct vs wrong)
- Consistency with baseline
- Comparison against baseline accuracy

Each track gets its own plot with 4 bars (2 aug modes × 2 variants).
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def plot_adversarial_track(
    track: str,
    model_name: str,
    results_dir: str,
    baseline_dir: str,
    plots_dir: str,
    save_pdf: bool = False,
):
    """Generate a grouped bar plot for one track showing all 4 conditions."""
    
    print(f"\n--- Plotting: {track.upper()} ---")
    
    # Load baseline
    baseline_path = os.path.join(baseline_dir, f"baseline_{model_name}_sakura-{track}.jsonl")
    baseline_results = load_results(baseline_path)
    baseline_metrics = compute_metrics(baseline_results)
    baseline_acc = baseline_metrics['accuracy'] * 100 if baseline_metrics['accuracy'] is not None else None
    
    # Load all 4 adversarial conditions
    conditions = [
        ('concat', 'correct', '#1f77b4'),
        ('concat', 'wrong', '#ff7f0e'),
        ('overlay', 'correct', '#2ca02c'),
        ('overlay', 'wrong', '#d62728'),
    ]
    
    accuracies = []
    consistencies = []
    labels = []
    colors = []
    
    for aug, variant, color in conditions:
        filename = f"adversarial_{model_name}_sakura-{track}_{aug}_{variant}.jsonl"
        filepath = os.path.join(results_dir, aug, filename)
        
        results = load_results(filepath)
        metrics = compute_metrics(results)
        
        if metrics['accuracy'] is not None:
            accuracies.append(metrics['accuracy'] * 100)
            consistencies.append(metrics['consistency'] * 100 if metrics['consistency'] is not None else 0)
            labels.append(f"{aug.capitalize()}\n{variant.capitalize()}")
            colors.append(color)
            print(f"  {aug}/{variant}: Acc={metrics['accuracy']*100:.1f}%, Consist={metrics['consistency']*100:.1f}%")
        else:
            print(f"  {aug}/{variant}: No data")
    
    if not accuracies:
        print(f"  No data found for {track}. Skipping plot.")
        return
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Accuracy plot
    bars1 = ax1.bar(x, accuracies, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    if baseline_acc is not None:
        ax1.axhline(y=baseline_acc, color='black', linestyle='--', linewidth=2, label=f'Baseline: {baseline_acc:.1f}%')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Adversarial Audio Accuracy\n{track.capitalize()} Track ({model_name.upper()})', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Consistency plot
    bars2 = ax2.bar(x, consistencies, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Consistency with Baseline (%)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Consistency with Baseline\n{track.capitalize()} Track ({model_name.upper()})', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(plots_dir, model_name, 'adversarial')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"adversarial_{model_name}_sakura-{track}"
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {png_path}")
    
    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"  ✓ Saved: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot adversarial experiment results per dataset.")
    parser.add_argument('--model', type=str, default='qwen', help="Model name")
    parser.add_argument('--tracks', nargs='+', default=['animal', 'emotion', 'gender', 'language'],
                        help="Tracks to plot")
    parser.add_argument('--results-dir', type=str, default='results/qwen/adversarial',
                        help="Adversarial results directory")
    parser.add_argument('--baseline-dir', type=str, default=None,
                        help="Baseline results directory (default: results/{model}/baseline)")
    parser.add_argument('--plots-dir', type=str, default='plots/per_dataset_plots',
                        help="Output directory for plots")
    parser.add_argument('--save-pdf', action='store_true', help="Save PDF copies")
    
    args = parser.parse_args()
    
    if args.baseline_dir is None:
        args.baseline_dir = f"results/{args.model}/baseline"
    
    print("=" * 60)
    print(f"Adversarial Audio Experiment Plotting ({args.model.upper()})")
    print("=" * 60)
    
    for track in args.tracks:
        plot_adversarial_track(
            track, args.model, args.results_dir, args.baseline_dir,
            args.plots_dir, args.save_pdf
        )
    
    print("\n" + "=" * 60)
    print("Plotting complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
