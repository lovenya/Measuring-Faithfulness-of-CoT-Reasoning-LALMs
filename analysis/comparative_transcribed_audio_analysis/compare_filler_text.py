# analysis/comparative_transcribed_audio_analysis/compare_filler_text.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from analysis.utils import load_results

def create_comparative_plot(dataset_name: str, results_dir: str, plots_dir: str):
    """
    Generates a single plot comparing the 'filler_text' results for the
    'default' and 'transcribed_audio' conditions.
    """
    print(f"\n--- Generating Comparative Filler Text Analysis for: {dataset_name.upper()} ---")
    
    try:
        # --- 1. Load Data for Both Conditions & Benchmarks ---
        # We load the results for the 'default' (original audio) condition.
        filler_default_df = load_results(results_dir, 'filler_text', dataset_name, 'default')
        
        # We load the results for the 'transcribed_audio' condition.
        filler_transcribed_df = load_results(results_dir, 'filler_text', dataset_name, 'transcribed_audio')
        
        # We only need the 'default' condition benchmarks for context.
        baseline_df = load_results(results_dir, 'baseline', dataset_name, 'default')
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name, 'default')
        
    except FileNotFoundError:
        print("  - Skipping analysis due to missing one or more required result files.")
        return

    # --- 2. Data Integrity: Inner Merge ---
    # To ensure a fair, "apples-to-apples" comparison, we only analyze the data points
    # (identified by question 'id' and 'percentile') that exist in BOTH result files.
    merge_keys = ['id', 'percentile']
    combined_df = pd.merge(filler_default_df, filler_transcribed_df, on=merge_keys, suffixes=('_default', '_transcribed'))

    if combined_df.empty:
        print("  - No common data points found between the two conditions. Skipping plot.")
        return

    # --- 3. Calculate Curves and Benchmarks ---
    # Calculate the two main "dose-response" curves from the merged, clean data.
    curve_default = combined_df.groupby('percentile')['is_correct_default'].mean() * 100
    curve_transcribed = combined_df.groupby('percentile')['is_correct_transcribed'].mean() * 100

    # Calculate context-aware benchmarks using only the questions present in our merged data.
    relevant_question_ids = combined_df[['id']].drop_duplicates()
    relevant_baseline_df = pd.merge(baseline_df, relevant_question_ids, on='id')
    relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_question_ids, on='id')
    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100

    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot the 'default' condition with a SOLID line.
    ax.plot(curve_default.index, curve_default.values, marker='o', linestyle='-', label='Accuracy (Original Audio)')
    
    # Plot the 'transcribed_audio' condition with a DASHED line.
    ax.plot(curve_transcribed.index, curve_transcribed.values, marker='o', linestyle='--', color='dodgerblue', label='Accuracy (Transcribed Audio)')

    # Plot the benchmark lines for context.
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Baseline CoT Accuracy ({baseline_accuracy:.2f}%)')

    # --- 5. Formatting and Saving ---
    ax.set_title(f'Filler Text Comparison: Original vs. Transcribed Audio ({dataset_name.upper()})', fontsize=16, pad=20)
    ax.set_xlabel('Percentage of CoT Replaced by Filler Text', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Condition', loc='best'); fig.tight_layout()

    # Save the plot to our standard comparative directory structure.
    output_plot_dir = os.path.join(plots_dir, 'comparative_transcribed_audio', 'filler_text', dataset_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"compare_filler_text_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparative Filler Text plots.")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        # This logic finds all datasets that have completed runs for BOTH conditions.
        try:
            results_subdir = os.path.join(args.results_dir, 'filler_text')
            datasets_default = set([f.replace('filler_text_', '').replace('_default.jsonl', '') for f in os.listdir(results_subdir) if f.endswith('_default.jsonl')])
            datasets_transcribed = set([f.replace('filler_text_', '').replace('_transcribed_audio.jsonl', '') for f in os.listdir(results_subdir) if f.endswith('_transcribed_audio.jsonl')])
            
            common_datasets = sorted(list(datasets_default & datasets_transcribed))
            print(f"Found common datasets for comparison: {common_datasets}")
            
            for dataset in common_datasets:
                create_comparative_plot(dataset, args.results_dir, args.plots_dir)
        except FileNotFoundError:
            print(f"Could not find 'filler_text' results directory at {results_subdir}.")
    else:
        create_comparative_plot(args.dataset, args.results_dir, args.plots_dir)