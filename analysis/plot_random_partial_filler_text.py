# analysis/plot_random_partial_filler_text.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Generates and saves a single plot for a given group of random partial filler text data.
    Correctly applies binning ONLY to the aggregated plot.
    """
    num_chains = len(df[['id', 'chain_id']].drop_duplicates())
    
    # --- Context-Aware Benchmark Calculation ---
    relevant_question_ids = df[['id']].drop_duplicates()
    relevant_baseline_df = pd.merge(baseline_df, relevant_question_ids, on='id')
    relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_question_ids, on='id')
    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    no_cot_accuracy = None
    if no_cot_df is not None:
        relevant_no_cot_df = pd.merge(no_cot_df, relevant_question_ids, on='id')
        if not relevant_no_cot_df.empty:
            no_cot_accuracy = relevant_no_cot_df.groupby('id')['is_correct'].mean().mean() * 100

    # --- Curve Generation with Conditional Binning ---
    if plot_group_name == 'aggregated':
        df['percent_binned'] = (df['percent_replaced'] / 5).round() * 5
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        accuracy_curve = df.groupby('percent_replaced')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_replaced')['is_consistent_with_baseline'].mean() * 100
    
    # Correctly define the 0% point for both curves.
    accuracy_curve[0] = baseline_accuracy
    consistency_curve[0] = 100.0
    accuracy_curve.sort_index(inplace=True)
    consistency_curve.sort_index(inplace=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy')
    ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Original Answer')

    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Accuracy & Consistency vs. Random CoT Corruption ({dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Random Reasoning Sentences Replaced by Filler', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, 'random_partial_filler_text', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, 'random_partial_filler_text', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"partial_filler_random_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


def create_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool, include_no_cot: bool):
    """ Main function to orchestrate the analysis. """
    print(f"\n--- Generating Random Partial Filler Analysis for: {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        partial_df = load_results(results_dir, 'random_partial_filler_text', dataset_name)
        early_df = load_results(results_dir, 'early_answering', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        print("  - Skipping plot due to missing one or more required result files.")
        return

    # --- Data Preparation ---
    # Add the 'is_consistent_with_baseline' column by merging with baseline predictions.
    baseline_predictions = baseline_df[['id', 'chain_id', 'predicted_choice']].rename(columns={'predicted_choice': 'baseline_predicted_choice'})
    combined_df = pd.merge(partial_df, baseline_predictions, on=['id', 'chain_id'], how='inner')
    combined_df['is_consistent_with_baseline'] = (combined_df['predicted_choice'] == combined_df['baseline_predicted_choice'])

    # Merge with early_answering data to get total_sentences_in_chain for grouping.
    sentence_counts = early_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
    combined_df = pd.merge(combined_df, sentence_counts, on=['id', 'chain_id'], how='inner')

    print("Generating main aggregated plot...")
    plot_single_graph(combined_df, baseline_df, no_reasoning_df, no_cot_df, 'aggregated', dataset_name, plots_dir)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = combined_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, no_cot_df, f'{total_steps}_sentences', dataset_name, plots_dir)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for random partial filler text.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to analyze ('mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--include-no-cot', action='store_true')
    parser.add_argument('--grouped', action='store_true', help='Generate detailed plots for each CoT length.')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            baseline_dir = os.path.join(args.results_dir, 'baseline')
            dataset_names = sorted([f.replace('baseline_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])
            for dataset in dataset_names:
                create_analysis(dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot)
        except FileNotFoundError:
            print(f"Could not find baseline directory at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        create_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot)