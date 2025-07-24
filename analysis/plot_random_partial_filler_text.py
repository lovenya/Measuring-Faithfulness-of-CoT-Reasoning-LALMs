# analysis/plot_random_partial_filler_text.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def create_plot(dataset_name: str, results_dir: str, plots_dir: str, include_no_cot: bool):
    """ Analyzes random partial filler text results. """
    print(f"\n--- Generating Random Partial Filler Analysis for: {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
        partial_df = load_results(results_dir, 'random_partial_filler_text', dataset_name)
    except FileNotFoundError:
        print("  - Skipping plot due to missing one or more required result files.")
        return

    baseline_accuracy = baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    no_cot_accuracy = no_cot_df.groupby('id')['is_correct'].mean().mean() * 100 if no_cot_df is not None else None

    accuracy_curve = partial_df.groupby('percent_replaced')['is_correct'].mean() * 100
    accuracy_curve[0] = baseline_accuracy
    accuracy_curve.sort_index(inplace=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='s', linestyle='-', label='Accuracy with Partial Filler')
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    ax.set_title(f'Accuracy vs. Random CoT Corruption ({dataset_name.upper()})', fontsize=16, pad=20)
    ax.set_xlabel('% of Random Reasoning Sentences Replaced by Filler', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Conditions', loc='best'); fig.tight_layout()

    output_plot_dir = os.path.join(plots_dir, 'random_partial_filler_text', dataset_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"partial_filler_random_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"Plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    # This script can be run with the same arguments as the others
    parser = argparse.ArgumentParser(description="Generate plots for random partial filler text.")
    
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to analyze ('mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--include-no-cot', action='store_true')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        # Logic to run for all datasets
        baseline_dir = os.path.join(args.results_dir, 'baseline')
        dataset_names = sorted([f.replace('baseline_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])
        for dataset in dataset_names:
            create_plot(dataset, args.results_dir, args.plots_dir, args.include_no_cot)
    else:
        create_plot(args.dataset, args.results_dir, args.plots_dir, args.include_no_cot)