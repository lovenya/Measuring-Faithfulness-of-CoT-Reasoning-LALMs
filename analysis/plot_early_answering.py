# analysis/plot_early_answering.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Generates and saves a single plot for a given group of early answering data.
    This function correctly applies data point binning ONLY to the main 'aggregated' plot.
    """
    # Calculate the number of unique reasoning chains included in this specific plot.
    num_chains = len(df[['id', 'chain_id']].drop_duplicates())
    
    # --- Benchmark Calculation ---
    # To ensure a fair comparison, benchmarks are calculated only on the subset of questions
    # that are actually present in the current data frame (df).
    relevant_question_ids = df[['id']].drop_duplicates()
    relevant_baseline_df = pd.merge(baseline_df, relevant_question_ids, on='id')
    relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_question_ids, on='id')
    
    # Use robust macro-averaging (average per question, then average the averages).
    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    
    no_cot_accuracy = None
    if no_cot_df is not None:
        relevant_no_cot_df = pd.merge(no_cot_df, relevant_question_ids, on='id')
        if not relevant_no_cot_df.empty:
            no_cot_accuracy = relevant_no_cot_df.groupby('id')['is_correct'].mean().mean() * 100

    # --- Curve Generation with Conditional Binning ---
    # This is a critical step for methodological correctness.
    if plot_group_name == 'aggregated':
        # For the aggregated plot, which combines CoTs of many different lengths,
        # we must bin the x-axis values into common buckets (e.g., 5%, 10%)
        # to create a single, coherent curve.
        df['percent_binned'] = (df['percent_reasoning_provided'] / 5).round() * 5
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        # For grouped plots, which only contain CoTs of a single length (e.g., 4 sentences),
        # the x-axis points are already precise and consistent (0%, 25%, 50%, etc.).
        # Binning is unnecessary and would distort the data, so we group by the raw percentages.
        accuracy_curve = df.groupby('percent_reasoning_provided')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_reasoning_provided')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    # Plot the two main curves with our standard aesthetic style.
    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy at Step')
    ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Final Answer')

    # Plot the horizontal benchmark lines for context.
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Final CoT Accuracy ({baseline_accuracy:.2f}%)')

    # Create a clear, two-line title that includes the chain count.
    base_title = f'Accuracy & Consistency vs. Reasoning Progression ({dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    # Set labels and limits for a clean, readable plot.
    ax.set_xlabel('% of Reasoning Chain Provided', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Save Figure to Organized Directory ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, 'early_answering', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, 'early_answering', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"early_answering_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


def create_early_answering_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool, include_no_cot: bool):
    """ Main function to orchestrate the early answering analysis. """
    print(f"\n--- Generating Early Answering Analysis for: {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        early_df = load_results(results_dir, 'early_answering', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        return

    # Filter out any trials where the sanitized CoT was empty.
    early_df = early_df[early_df['total_sentences_in_chain'] > 0].copy()
    if early_df.empty:
        print("No valid early answering data to plot (all CoTs might have been empty).")
        return

    # Calculate the percentage of the CoT provided at each step. This is our primary x-axis variable.
    early_df['percent_reasoning_provided'] = (early_df['num_sentences_provided'] / early_df['total_sentences_in_chain']) * 100

    # Always generate the main, high-level aggregated plot.
    print("Generating main aggregated plot...")
    plot_single_graph(early_df, baseline_df, no_reasoning_df, no_cot_df, 'aggregated', dataset_name, plots_dir)

    # If requested, generate the more detailed per-length plots.
    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = early_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            # To avoid noisy plots, only generate a grouped plot if there's a sufficient
            # number of data points (e.g., more than 10 unique chains).
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, no_cot_df, f'{total_steps}_sentences', dataset_name, plots_dir)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Early Answering plots for LALM results.")
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
                create_early_answering_analysis(dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot)
        except FileNotFoundError:
            print(f"Could not find baseline directory at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        create_early_answering_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot)