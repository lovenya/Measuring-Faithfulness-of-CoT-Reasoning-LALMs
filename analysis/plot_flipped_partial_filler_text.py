# analysis/plot_flipped_partial_filler_text.py

"""
This script generates plots for the "Flipped Partial Filler Text" experiment.

The scientific goal is to investigate whether the *position* of information in
a reasoning chain is critical to the model's performance. Specifically, it tests
if the model relies more heavily on the initial premises or the concluding steps
of its reasoning.

To achieve this, we replace a percentage of words in a valid reasoning chain with
meaningless filler text, but we do so starting from the *end* of the chain and
working backwards.

By comparing the results of this experiment to the standard "corruption from
start" experiment, we can infer the structural importance of different parts of
the reasoning process.

The script produces aggregated and per--CoT-length (grouped) plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, plot_group_name: str, model_name: str, dataset_name: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_accuracy: bool, show_consistency: bool, show_baseline: bool, show_nr: bool):
    """
    Generates and saves a single plot for a given group of 'Flipped Partial Filler' data.

    Args:
        df (pd.DataFrame): The main DataFrame for the plot.
        baseline_df (pd.DataFrame): DataFrame with baseline results for benchmarks.
        no_reasoning_df (pd.DataFrame): DataFrame with no-reasoning results for benchmarks.
        plot_group_name (str): The name of the group being plotted (e.g., 'aggregated').
        model_name (str): The name of the model being analyzed.
        dataset_name (str): The name of the dataset being analyzed.
        plots_dir (str): The root directory to save the plots in.
        is_restricted (bool): Flag indicating if the analysis is on the restricted dataset.
        save_as_pdf (bool): If True, saves a PDF copy of the plot.
        show_accuracy (bool): If True, plots the accuracy curve.
        show_consistency (bool): If True, plots the consistency curve.
        show_baseline (bool): If True, plots the baseline benchmark line.
        show_nr (bool): If True, plots the no-reasoning benchmark line.
    """
    num_chains = len(df[['id', 'chain_id']].drop_duplicates())

    # --- Benchmark Calculation ---
    if plot_group_name == 'aggregated':
        relevant_ids = df[['id']].drop_duplicates()
        relevant_baseline_df = pd.merge(baseline_df, relevant_ids, on='id')
        relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_ids, on='id')
    else:
        relevant_ids = df[['id', 'chain_id']].drop_duplicates()
        relevant_baseline_df = pd.merge(baseline_df, relevant_ids, on=['id', 'chain_id'])
        relevant_no_reasoning_df = pd.merge(no_reasoning_df, df[['id']].drop_duplicates(), on='id')

    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    
    # --- Curve Generation with Conditional Binning ---
    if plot_group_name == 'aggregated':
        df['percent_binned'] = (df['percent_replaced'] / 5).round() * 5
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        accuracy_curve = df.groupby('percent_replaced')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_replaced')['is_consistent_with_baseline'].mean() * 100
    
    # Synthetically add the 0% data point.
    accuracy_curve[0] = baseline_accuracy
    consistency_curve[0] = 100.0
    accuracy_curve.sort_index(inplace=True)
    consistency_curve.sort_index(inplace=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    if show_accuracy:
        ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy')
    if show_consistency:
        ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Original Answer')
    if show_nr:
        ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if show_baseline:
        ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    restriction_str = " (Restricted)" if is_restricted else " (Full Dataset)"
    base_title = f'Accuracy & Consistency vs. CoT Corruption from End ({model_name.upper()} on {dataset_name.upper()}){restriction_str}'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Final Reasoning Replaced by Filler', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Output Path Construction ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, model_name, 'flipped_partial_filler_text', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, model_name, 'flipped_partial_filler_text', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"partial_filler_end_{model_name}_{dataset_name}_{plot_group_name}{suffix}"
    
    png_path = os.path.join(output_plot_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_plot_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()

def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, is_restricted: bool, generate_grouped: bool, save_as_pdf: bool, show_flags: dict):
    """
    Main function to orchestrate the analysis for a single dataset.
    """
    print(f"\n--- Generating Flipped Partial Filler Analysis for: {model_name.upper()} on {dataset_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name, is_restricted)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name, is_restricted)
        partial_df = load_results(model_name, results_dir, 'flipped_partial_filler_text', dataset_name, is_restricted)
        # We must load early_answering to get the total sentence counts for grouping.
        early_df = load_results(model_name, results_dir, 'early_answering', dataset_name, is_restricted)
    except FileNotFoundError:
        return

    # --- Data Preparation ---
    baseline_predictions = baseline_df[['id', 'chain_id', 'predicted_choice']].rename(columns={'predicted_choice': 'baseline_predicted_choice'})
    combined_df = pd.merge(partial_df, baseline_predictions, on=['id', 'chain_id'], how='inner')
    combined_df['is_consistent_with_baseline'] = (combined_df['predicted_choice'] == combined_df['baseline_predicted_choice'])

    sentence_counts = early_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
    combined_df = pd.merge(combined_df, sentence_counts, on=['id', 'chain_id'], how='inner')
    
    # --- "Meaningful Manipulation" Filter ---
    combined_df = combined_df[combined_df['total_sentences_in_chain'] > 0].copy()
    if combined_df.empty:
        print("  - No valid data with non-empty CoTs found. Skipping analysis.")
        return

    print("Generating main aggregated plot...")
    plot_single_graph(combined_df, baseline_df, no_reasoning_df, 'aggregated', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = combined_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, f'{total_steps}_sentences', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data (<=10 chains).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for flipped partial filler text.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--restricted', action='store_true')
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--save-pdf', action='store_true')
    # "Opt-In" flags for controlling which elements are visible on the plot.
    parser.add_argument('--show-accuracy-curve', action='store_true')
    parser.add_argument('--show-consistency-curve', action='store_true')
    parser.add_argument('--show-baseline-benchmark', action='store_true')
    parser.add_argument('--show-nr-benchmark', action='store_true')
    args = parser.parse_args()
    
    show_flags = {
        "show_accuracy": args.show_accuracy_curve,
        "show_consistency": args.show_consistency_curve,
        "show_baseline": args.show_baseline_benchmark,
        "show_nr": args.show_nr_benchmark
    }

    if args.dataset == 'all':
        try:
            baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            if args.restricted:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
            else:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if not f.endswith('-restricted.jsonl')])))
            
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at {baseline_dir}.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)