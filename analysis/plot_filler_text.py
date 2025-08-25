# analysis/plot_filler_text.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from .utils import load_results

def create_filler_text_plot(dataset_name: str, results_dir: str, plots_dir: str, include_no_cot: bool):
    """
    Analyzes filler text results, generating a plot comparing accuracy against
    context-aware, macro-averaged benchmarks.
    """
    print(f"\n--- Generating Filler Text Analysis for: {dataset_name.upper()} ---")
    
    try:
        # Load all necessary result files.
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        filler_df = load_results(results_dir, 'filler_text', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        return

    # --- Context-Aware Benchmark Calculation ---
    # This is a critical step for a fair, "apples-to-apples" comparison.
    # We identify the unique questions present in the filler_text results,
    # and then filter all benchmark dataframes to only include those same questions.
    relevant_question_ids = filler_df[['id']].drop_duplicates()
    
    relevant_baseline_df = pd.merge(baseline_df, relevant_question_ids, on='id')
    relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_question_ids, on='id')

    # Calculate benchmark accuracies using robust macro-averaging on the filtered data.
    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    
    no_cot_accuracy = None
    if no_cot_df is not None:
        relevant_no_cot_df = pd.merge(no_cot_df, relevant_question_ids, on='id')
        if not relevant_no_cot_df.empty:
            no_cot_accuracy = relevant_no_cot_df.groupby('id')['is_correct'].mean().mean() * 100

    # --- Curve Generation ---
    # The main curve is calculated by grouping the filler text results by the
    # percentage of the CoT that was replaced.
    filler_accuracy_by_percentile = filler_df.groupby('percentile')['is_correct'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    # Plot the main "dose-response" curve.
    ax.plot(filler_accuracy_by_percentile.index, filler_accuracy_by_percentile.values, 
            marker='o', linestyle='-', label='Filler Text Accuracy')

    # Plot the context-aware benchmark lines.
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Baseline CoT Accuracy ({baseline_accuracy:.2f}%)')

    # --- Formatting and Saving ---
    ax.set_title(f'Impact of CoT Content vs. Compute Time ({dataset_name.upper()})', fontsize=16, pad=20)
    ax.set_xlabel('Percentage of CoT Replaced by Filler Text', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Conditions', loc='best'); fig.tight_layout()

    output_plot_dir = os.path.join(plots_dir, 'filler_text', dataset_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"filler_text_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"Plot saved successfully to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Filler Text plots for LALM results.")
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
                create_filler_text_plot(dataset, args.results_dir, args.plots_dir, args.include_no_cot)
        except FileNotFoundError:
            print(f"Could not find baseline directory at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        create_filler_text_plot(args.dataset, args.results_dir, args.plots_dir, args.include_no_cot)