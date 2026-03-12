# analysis/plot_early_answering.py

"""
This script generates plots for the "Early Answering" experiment.

The scientific goal of this experiment is to probe for post-hoc reasoning by
observing when the model's final answer "locks in." We do this by presenting
the model with progressively longer portions of a reasoning chain (from 0% to 100%)
and recording its answer at each step.

The script produces two types of plots:
1. Aggregated Plots: A high-level view showing the average trend across all
   reasoning chains for a given dataset.
2. Grouped Plots: Detailed, per-CoT-length plots that provide a more granular
   view of the model's behavior for reasoning of a specific length (e.g., all
   4-sentence chains).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, plot_group_name: str, model_name: str, dataset_name: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_accuracy: bool, show_consistency: bool, show_baseline: bool, show_nr: bool):
    """
    Generates and saves a single plot for a given group of 'Early Answering' data.

    Args:
        df (pd.DataFrame): The main DataFrame containing the early answering data for the plot.
        baseline_df (pd.DataFrame): DataFrame with full baseline results for benchmark calculation.
        no_reasoning_df (pd.DataFrame): DataFrame with no-reasoning results for benchmark calculation.
        plot_group_name (str): The name of the group being plotted (e.g., 'aggregated', '4_sentences').
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
    # Calculate the number of unique reasoning chains included in this specific plot.
    num_chains = len(df[['id', 'chain_id']].drop_duplicates())

    # --- Benchmark Calculation ---
    # This block calculates the benchmark accuracies. The methodology differs for
    # aggregated vs. grouped plots to ensure the most scientifically valid comparison.
    if plot_group_name == 'aggregated':
        # For the main aggregated plot, we use the "Question-Level" benchmark.
        # This compares performance against the average for all chains of the relevant questions.
        relevant_ids = df[['id']].drop_duplicates()
        relevant_baseline_df = pd.merge(baseline_df, relevant_ids, on='id')
        relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_ids, on='id')
    else:
        # For grouped plots, we use the more direct "Chain-Level" benchmark.
        # This compares performance against only the *specific chains* in this group.
        relevant_ids = df[['id', 'chain_id']].drop_duplicates()
        relevant_baseline_df = pd.merge(baseline_df, relevant_ids, on=['id', 'chain_id'])
        # The no_reasoning benchmark is always question-level as there's no chain-to-chain link.
        relevant_no_reasoning_df = pd.merge(no_reasoning_df, df[['id']].drop_duplicates(), on='id')

    # The calculation itself is a robust macro-average.
    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    
    # --- Curve Generation with Conditional Binning ---
    # This is a critical methodological step.
    if plot_group_name == 'aggregated':
        # Binning is necessary for the aggregated plot to create a common, smooth x-axis
        # from CoTs of many different lengths.
        df['percent_binned'] = (df['percent_reasoning_provided'] / 5).round() * 5
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        # Grouped plots use the raw, precise percentages for maximum data fidelity.
        # Binning here would distort the results.
        accuracy_curve = df.groupby('percent_reasoning_provided')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_reasoning_provided')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    # Conditionally plot each element based on the "opt-in" flags.
    if show_accuracy:
        ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy at Step')
    if show_consistency:
        ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Final Answer')
    if show_nr:
        ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if show_baseline:
        ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Final CoT Accuracy ({baseline_accuracy:.2f}%)')

    # Create a clear, multi-line title that includes all relevant context.
    restriction_str = " (Restricted)" if is_restricted else " (Full Dataset)"
    base_title = f'Accuracy & Consistency vs. Reasoning Progression ({model_name.upper()} on {dataset_name.upper()}){restriction_str}'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Provided', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Output Path Construction ---
    # The path is constructed to be model- and restriction-aware.
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, model_name, 'early_answering', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, model_name, 'early_answering', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"early_answering_{model_name}_{dataset_name}_{plot_group_name}{suffix}"
    
    # Save the PNG and, optionally, the PDF.
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
    Main function to orchestrate the early answering analysis for a single dataset.
    It loads the data, performs necessary calculations, and calls the plotting function.
    """
    print(f"\n--- Generating Early Answering Analysis for: {model_name.upper()} on {dataset_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    try:
        # Load all necessary data files using our centralized utility.
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name, is_restricted)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name, is_restricted)
        early_df = load_results(model_name, results_dir, 'early_answering', dataset_name, is_restricted)
    except FileNotFoundError:
        # If any required file is missing, we cannot proceed.
        return
    
    # --- "Meaningful Manipulation" Filter ---
    # This experiment is only valid for chains that have at least one sentence.
    # We filter out the zero-step chains to ensure the analysis is scientifically valid.
    early_df = early_df[early_df['total_sentences_in_chain'] > 0].copy()
    if early_df.empty:
        print("  - No valid data with non-empty CoTs found. Skipping analysis.")
        return

    # Calculate the normalized x-axis value for each step.
    early_df['percent_reasoning_provided'] = (early_df['num_sentences_provided'] / early_df['total_sentences_in_chain']) * 100

    # Generate the main, high-level aggregated plot.
    print("Generating main aggregated plot...")
    plot_single_graph(early_df, baseline_df, no_reasoning_df, 'aggregated', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)

    # If requested, generate the more detailed per-length plots.
    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = early_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            # To avoid noisy plots, only generate a grouped plot if there's a sufficient
            # number of data points (e.g., more than 10 unique chains).
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, f'{total_steps}_sentences', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data (<=10 chains).")


if __name__ == "__main__":
    # This block handles parsing the command-line arguments and orchestrating the analysis.
    parser = argparse.ArgumentParser(description="Generate Early Answering plots for LALM results.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen', 'salmonn').")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar', or 'all' for batch processing).")
    parser.add_argument('--results_dir', type=str, default='./results', help="The root directory where experiment results are stored.")
    parser.add_argument('--plots_dir', type=str, default='./plots', help="The directory where generated plots will be saved.")
    parser.add_argument('--restricted', action='store_true', help="Analyze the '-restricted.jsonl' files.")
    parser.add_argument('--grouped', action='store_true', help="Generate detailed plots for each CoT length in addition to the main aggregated plot.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of each plot.")
    
    # "Opt-In" flags for controlling which elements are visible on the plot.
    parser.add_argument('--show-accuracy-curve', action='store_true', help="Display the accuracy curve on the plot.")
    parser.add_argument('--show-consistency-curve', action='store_true', help="Display the consistency curve on the plot.")
    parser.add_argument('--show-baseline-benchmark', action='store_true', help="Display the baseline accuracy benchmark line.")
    parser.add_argument('--show-nr-benchmark', action='store_true', help="Display the no-reasoning accuracy benchmark line.")
    
    args = parser.parse_args()
    
    # Pack the "show" flags into a dictionary to be passed to the plotting function.
    show_flags = {
        "show_accuracy": args.show_accuracy_curve,
        "show_consistency": args.show_consistency_curve,
        "show_baseline": args.show_baseline_benchmark,
        "show_nr": args.show_nr_benchmark
    }

    # This logic handles the '--dataset all' batch processing mode.
    if args.dataset == 'all':
        try:
            # Discover available datasets from the baseline directory, our ground truth.
            baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            if args.restricted:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
            else:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if not f.endswith('-restricted.jsonl')])))
            
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            # Loop through the discovered datasets and run the analysis for each one.
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        # If a single dataset is specified, just run the analysis for that one.
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)
