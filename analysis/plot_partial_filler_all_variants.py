# analysis/plot_partial_filler_variants.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def create_partial_filler_plot(dataset_name: str, results_dir: str, plots_dir: str, include_no_cot: bool):
    """
    Analyzes all three partial filler text variants and plots them on a single graph for comparison.
    """
    print(f"\n--- Generating Combined Partial Filler Analysis for: {dataset_name.upper()} ---")
    
    # 1. Load all necessary data
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
        
        # Load the three variant datasets
        start_df = load_results(results_dir, 'partial_filler_text', dataset_name)
        flipped_df = load_results(results_dir, 'flipped_partial_filler_text', dataset_name)
        random_df = load_results(results_dir, 'random_partial_filler_text', dataset_name)
        
    except FileNotFoundError:
        print("  - Skipping plot due to missing one or more required result files.")
        return

    # 2. Calculate benchmark accuracies using macro-averaging
    baseline_accuracy = baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    no_cot_accuracy = no_cot_df.groupby('id')['is_correct'].mean().mean() * 100 if no_cot_df is not None else None

    # 3. Calculate the accuracy curves for each variant
    start_curve = start_df.groupby('percent_replaced')['is_correct'].mean() * 100
    flipped_curve = flipped_df.groupby('percent_replaced')['is_correct'].mean() * 100
    random_curve = random_df.groupby('percent_replaced')['is_correct'].mean() * 100
    
    # Methodologically, the 0% point for all curves is the baseline accuracy.
    start_curve[0] = flipped_curve[0] = random_curve[0] = baseline_accuracy
    start_curve.sort_index(inplace=True)
    flipped_curve.sort_index(inplace=True)
    random_curve.sort_index(inplace=True)

    # 4. Generate the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    # Plot the three variant curves with distinct styles
    ax.plot(start_curve.index, start_curve.values, marker='o', linestyle='-', label='Corruption from Start')
    ax.plot(flipped_curve.index, flipped_curve.values, marker='^', linestyle='-', label='Corruption from End')
    ax.plot(random_curve.index, random_curve.values, marker='s', linestyle='-', label='Random Corruption')

    # Plot the benchmark lines
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT (Freeflow) Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    # 5. Formatting
    ax.set_title(f'Accuracy vs. Positional CoT Corruption ({dataset_name.upper()})', fontsize=16, pad=20)
    ax.set_xlabel('% of CoT Sentences Replaced by Filler', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.legend(title='Corruption Method', loc='best')
    fig.tight_layout()

    # 6. Save the figure to the correct, organized directory
    output_plot_dir = os.path.join(plots_dir, 'partial_filler_all_variants', dataset_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"partial_filler_all_variants_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate combined Partial Filler Text plots for LALM results.")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results', help="The root directory where experiment results are stored.")
    parser.add_argument('--plots_dir', type=str, default='./plots', help="The directory where generated plots will be saved.")
    parser.add_argument('--include-no-cot', action='store_true', help='Also load and plot the No-CoT (freeflow) results as a benchmark.')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            baseline_dir = os.path.join(args.results_dir, 'baseline')
            dataset_names = sorted([f.replace('baseline_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])
            print(f"Found datasets: {dataset_names}")
            for dataset in dataset_names:
                create_partial_filler_plot(dataset, args.results_dir, args.plots_dir, args.include_no_cot)
        except FileNotFoundError:
            print(f"Could not find baseline directory at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        create_partial_filler_plot(args.dataset, args.results_dir, args.plots_dir, args.include_no_cot)