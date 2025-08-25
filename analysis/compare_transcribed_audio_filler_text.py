# analysis/compare_transcribed_audio_filler_text.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from .utils import load_results

def calculate_accuracy(df: pd.DataFrame, pred_col: str, correct_col: str) -> float:
    """
    Calculates accuracy only on the subset of trials where the model provided a valid answer (A, B, C, etc.).
    """
    valid_answers_df = df.dropna(subset=[pred_col])
    valid_answers_df = valid_answers_df[valid_answers_df[pred_col] != "REFUSAL"]
    if valid_answers_df.empty: return 0.0
    return (valid_answers_df[pred_col] == valid_answers_df[correct_col]).mean() * 100

def create_comparative_plot(dataset_name: str, results_dir: str, plots_dir: str):
    """
    Generates a single plot comparing the 'filler_text' results for the
    'default' and 'transcribed_audio' conditions.
    """
    print(f"\n--- Generating Comparative Filler Text Analysis for: {dataset_name.upper()} ---")
    
    try:
        # --- 1. Load Data for Both Conditions & Benchmarks ---
        filler_default_df = load_results(results_dir, 'filler_text', dataset_name, 'default')
        filler_transcribed_df = load_results(results_dir, 'filler_text', dataset_name, 'transcribed_audio')
        
        baseline_default_df = load_results(results_dir, 'baseline', dataset_name, 'default')
        baseline_transcribed_df = load_results(results_dir, 'baseline', dataset_name, 'transcribed_audio')
        no_reasoning_default_df = load_results(results_dir, 'no_reasoning', dataset_name, 'default')
        no_reasoning_transcribed_df = load_results(results_dir, 'no_reasoning', dataset_name, 'transcribed_audio')
        
    except FileNotFoundError:
        return

    # --- 2. Data Integrity: Inner Merge ---
    # We only analyze data points (question 'id' and 'percentile') that exist in BOTH result files.
    merge_keys = ['id', 'percentile']
    combined_df = pd.merge(filler_default_df, filler_transcribed_df, on=merge_keys, suffixes=('_default', '_transcribed'))

    if combined_df.empty:
        print("  - No common data points found between the two conditions. Skipping plot.")
        return

    # --- 3. Calculate Curves and Benchmarks ---
    # Calculate accuracy curves using our robust, null-excluding function.
    curve_default = combined_df.groupby('percentile').apply(lambda g: calculate_accuracy(g, 'predicted_choice_default', 'correct_choice_default'), include_groups=False)
    curve_transcribed = combined_df.groupby('percentile').apply(lambda g: calculate_accuracy(g, 'predicted_choice_transcribed', 'correct_choice_transcribed'), include_groups=False)

    # Calculate context-aware benchmarks.
    relevant_question_ids = combined_df[['id']].drop_duplicates()
    
    rel_bl_def = pd.merge(baseline_default_df, relevant_question_ids, on='id')
    rel_bl_trn = pd.merge(baseline_transcribed_df, relevant_question_ids, on='id')
    rel_nr_def = pd.merge(no_reasoning_default_df, relevant_question_ids, on='id')
    rel_nr_trn = pd.merge(no_reasoning_transcribed_df, relevant_question_ids, on='id')

    bench_bl_def_acc = calculate_accuracy(rel_bl_def, 'predicted_choice', 'correct_choice')
    bench_bl_trn_acc = calculate_accuracy(rel_bl_trn, 'predicted_choice', 'correct_choice')
    bench_nr_def_acc = calculate_accuracy(rel_nr_def, 'predicted_choice', 'correct_choice')
    bench_nr_trn_acc = calculate_accuracy(rel_nr_trn, 'predicted_choice', 'correct_choice')

    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot the two main "dose-response" curves.
    ax.plot(curve_default.index, curve_default.values, marker='o', linestyle='-', label='Accuracy (Original Audio)')
    ax.plot(curve_transcribed.index, curve_transcribed.values, marker='o', linestyle='--', color='dodgerblue', label='Accuracy (Transcribed Audio)')

    # Plot the four benchmark lines for a rich comparison.
    ax.axhline(y=bench_nr_def_acc, color='red', linestyle=':', label=f'No-Reasoning Acc (Original) ({bench_nr_def_acc:.2f}%)')
    ax.axhline(y=bench_nr_trn_acc, color='salmon', linestyle=':', label=f'No-Reasoning Acc (Transcribed) ({bench_nr_trn_acc:.2f}%)')
    ax.axhline(y=bench_bl_def_acc, color='green', linestyle='--', label=f'Baseline Acc (Original) ({bench_bl_def_acc:.2f}%)')
    ax.axhline(y=bench_bl_trn_acc, color='lime', linestyle='--', label=f'Baseline Acc (Transcribed) ({bench_bl_trn_acc:.2f}%)')

    # --- 5. Formatting and Saving ---
    ax.set_title(f'Filler Text Comparison: Original vs. Transcribed Audio ({dataset_name.upper()})', fontsize=16, pad=20)
    ax.set_xlabel('Percentage of CoT Replaced by Filler Text', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Condition', loc='best'); fig.tight_layout()

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
            default_dir = os.path.join(args.results_dir, 'filler_text')
            transcribed_dir = os.path.join(args.results_dir, 'transcribed_audio_experiments', 'filler_text')
            
            datasets_default, datasets_transcribed = set(), set()
            if os.path.exists(default_dir):
                datasets_default = set([f.replace('filler_text_', '').replace('.jsonl', '') for f in os.listdir(default_dir) if f.endswith('.jsonl')])
            if os.path.exists(transcribed_dir):
                datasets_transcribed = set([f.replace('filler_text_', '').replace('_transcribed_audio.jsonl', '') for f in os.listdir(transcribed_dir) if f.endswith('_transcribed_audio.jsonl')])
            
            common_datasets = sorted(list(datasets_default & datasets_transcribed))
            print(f"Found common datasets for comparison: {common_datasets}")
            
            for dataset in common_datasets:
                create_comparative_plot(dataset, args.results_dir, args.plots_dir)
        except FileNotFoundError:
            print(f"Could not find one of the required 'filler_text' results directories.")
    else:
        create_comparative_plot(args.dataset, args.results_dir, args.plots_dir)