# analysis/plot_adding_mistakes.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Helper function to generate a single 'Adding Mistakes' plot.
    This version correctly applies binning ONLY to the aggregated plot.
    """
    # --- Calculate N for the title ---
    num_chains = len(df[['id', 'chain_id']].drop_duplicates())

    # --- Macro-Averaging for Benchmarks ---
    # (This section is unchanged)
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

    # --- THE FIX: CONDITIONAL BINNING ---
    # First, filter the data as we agreed
    df_filtered = df[df['percent_before_mistake'] <= 90].copy()
    if df_filtered.empty:
        print(f"  - Skipping plot for '{plot_group_name}' as no data remains after filtering >90%.")
        return

    if plot_group_name == 'aggregated':
        # For the aggregated plot, we MUST bin to create a common x-axis.
        df_filtered['percent_binned'] = (df_filtered['percent_before_mistake'] / 10).round() * 10
        accuracy_curve = df_filtered.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df_filtered.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        # For grouped plots, we use the raw, precise percentages. NO BINNING.
        accuracy_curve = df_filtered.groupby('percent_before_mistake')['is_correct'].mean() * 100
        consistency_curve = df_filtered.groupby('percent_before_mistake')['is_consistent_with_baseline'].mean() * 100
    # --- END OF FIX ---

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_curve.index, accuracy_curve.values, 
            marker='^', linestyle='--', label='Accuracy After Mistake')
    ax.plot(consistency_curve.index, consistency_curve.values, 
            marker='o', linestyle='-', color='#8c564b', label='Consistency with Original Answer')

    # (The rest of the plotting code is unchanged)
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Accuracy & Consistency vs. Position of Introduced Mistake ({dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Before Mistake', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 95); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, 'adding_mistakes', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, 'adding_mistakes', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"adding_mistakes_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")
       
    
def create_adding_mistakes_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool, include_no_cot: bool):
    """ Main function to orchestrate the 'Adding Mistakes' analysis. """
    print(f"\n--- Generating Adding Mistakes Analysis for: {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        mistakes_df = load_results(results_dir, 'adding_mistakes', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        return

    if mistakes_df.empty:
        print("No valid 'Adding Mistakes' data to plot.")
        return

    mistakes_df['percent_before_mistake'] = ((mistakes_df['mistake_position'] - 1) / mistakes_df['total_sentences_in_chain']) * 100

    # 1. Always generate the main, aggregated plot
    print("Generating main aggregated plot...")
    plot_single_graph(mistakes_df, baseline_df, no_reasoning_df, no_cot_df, 'aggregated', dataset_name, plots_dir)

    # 2. Optionally, generate per-length grouped plots
    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = mistakes_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            # Check for a meaningful number of chains to plot
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, no_cot_df, f'{total_steps}_sentences', dataset_name, plots_dir)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Adding Mistakes plots for LALM results.")
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
                create_adding_mistakes_analysis(dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot)
        except FileNotFoundError:
            print(f"Could not find baseline directory at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        create_adding_mistakes_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot)