# analysis/plot_paraphrasing.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Helper function to generate a single paraphrasing plot, now with chain count in the title.
    """
    # --- NEW: Calculate the number of unique chains in this data group ---
    num_chains = len(df[['id', 'chain_id']].drop_duplicates())

    # --- Macro-Averaging for Benchmarks ---
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

    # --- Binning and Averaging for the Curves ---
    df['percent_binned'] = (df['percent_paraphrased'] / 5).round() * 5
    accuracy_by_step = df.groupby('percent_binned')['is_correct'].mean() * 100
    consistency_by_step = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_by_step.index, accuracy_by_step.values, marker='^', linestyle='--', label='Accuracy of Paraphrased CoT')
    ax.plot(consistency_by_step.index, consistency_by_step.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Original CoT')

    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle=':', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    # --- UPDATED TITLE LOGIC ---
    base_title = f'Accuracy & Consistency vs. Paraphrased Reasoning ({dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
    # --- END OF UPDATE ---
        
    ax.set_xlabel('% of Reasoning Chain Paraphrased', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Save Figure ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, 'paraphrasing', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, 'paraphrasing', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"paraphrasing_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")

def create_paraphrasing_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool, include_no_cot: bool):
    """ Main function to orchestrate the paraphrasing analysis. """
    print(f"\n--- Generating Paraphrasing Analysis for: {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        paraphrasing_df = load_results(results_dir, 'paraphrasing', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        return

    # --- Data Preparation ---
    # Create the 0% paraphrased data points from the baseline results
    baseline_for_concat = baseline_df.copy()
    baseline_for_concat['num_sentences_paraphrased'] = 0
    
    # We need total_sentences_in_chain to normalize. We can get this from any paraphrasing trial for a given chain.
    sentence_counts = paraphrasing_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
    if sentence_counts.empty:
        print("Paraphrasing results are empty or invalid. Cannot determine sentence counts.")
        return
        
    baseline_for_concat = pd.merge(baseline_for_concat, sentence_counts, on=['id', 'chain_id'], how='inner')
    
    # Combine the 0% (baseline) and >0% (paraphrasing) data
    combined_df = pd.concat([baseline_for_concat, paraphrasing_df], ignore_index=True, sort=False)
    combined_df = combined_df.dropna(subset=['total_sentences_in_chain'])
    combined_df['total_sentences_in_chain'] = combined_df['total_sentences_in_chain'].astype(int)
    combined_df = combined_df[combined_df['total_sentences_in_chain'] > 0]
    
    # Normalize the x-axis
    combined_df['percent_paraphrased'] = (combined_df['num_sentences_paraphrased'] / combined_df['total_sentences_in_chain']) * 100

    # --- Plotting ---
    print("Generating main aggregated plot...")
    plot_single_graph(combined_df, baseline_df, no_reasoning_df, no_cot_df, 'aggregated', dataset_name, plots_dir)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = combined_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df['chain_id'].unique()) * len(group_df['id'].unique()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, no_cot_df, f'{total_steps}_sentences', dataset_name, plots_dir)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Paraphrasing plots for LALM results.")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results', help="The root directory where experiment results are stored.")
    parser.add_argument('--plots_dir', type=str, default='./plots', help="The directory where generated plots will be saved.")
    parser.add_argument('--grouped', action='store_true', help='Generate detailed plots for each CoT length in addition to the main aggregated plot.')
    parser.add_argument('--include-no-cot', action='store_true', help='Also load and plot the No-CoT (freeflow) results as a benchmark.')
    args = parser.parse_args()
    
    
    if args.dataset == 'all':
        try:
            baseline_dir = os.path.join(args.results_dir, 'baseline')
            dataset_names = sorted([f.replace('baseline_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])
            print(f"Found datasets: {dataset_names}")
            for dataset in dataset_names:
                create_paraphrasing_analysis(dataset, args.results_dir, args.plots_dir, args.include_no_cot)
        except FileNotFoundError:
            print(f"Could not find baseline directory at {baseline_dir}. Cannot run for 'all' datasets.")
            
    else:
        create_paraphrasing_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot)