# analysis/plot_filler_text.py

"""
This script generates plots for the main "Filler Text" experiment.

The scientific goal is to disentangle the effect of reasoning *content* from the
effect of computational *length* (i.e., 'thinking time'). We test whether the
model's performance relies on the semantic meaning of its reasoning or simply
the sequence length it is allowed to process.

To achieve this, we use a percentile-based system. For each question, we
determine the token length of its longest reasoning chain from the baseline
results. We then measure the model's accuracy at 5% intervals (from 0% to 100%),
where each step involves replacing that percentage of the reasoning with a
token-equivalent amount of meaningless filler text ("...").

This manipulation is designed to keep the computational cost nearly identical at
each step while systematically destroying the semantic content.

This script produces an aggregated plot showing how accuracy degrades as a
function of the percentage of reasoning replaced.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, model_name: str, dataset_name: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_baseline: bool, show_nr: bool):
    """
    Generates and saves a single plot for the 'Filler Text' experiment.

    Args:
        df (pd.DataFrame): The main DataFrame containing the filler text data.
        baseline_df (pd.DataFrame): DataFrame with baseline results for benchmarks.
        no_reasoning_df (pd.DataFrame): DataFrame with no-reasoning results for benchmarks.
        model_name (str): The name of the model being analyzed.
        dataset_name (str): The name of the dataset being analyzed.
        plots_dir (str): The root directory to save the plots in.
        is_restricted (bool): Flag indicating if the analysis is on the restricted dataset.
        save_as_pdf (bool): If True, saves a PDF copy of the plot.
        show_baseline (bool): If True, plots the baseline benchmark line.
        show_nr (bool): If True, plots the no-reasoning benchmark line.
    """
    # The number of chains is calculated from the unique questions in the filler_text df.
    num_questions = len(df['id'].unique())
    
    # --- Context-Aware Benchmark Calculation (Question-Level) ---
    relevant_question_ids = df[['id']].drop_duplicates()
    relevant_baseline_df = pd.merge(baseline_df, relevant_question_ids, on='id')
    relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_question_ids, on='id')
    
    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    
    # --- Curve Generation ---
    # The main curve is the accuracy at each percentile of filler text.
    accuracy_curve = df.groupby('percentile')['is_correct'].mean() * 100
    
    # The 0% point is explicitly set to the no_reasoning accuracy, as per the experiment's design.
    accuracy_curve[0] = no_reasoning_accuracy
    accuracy_curve.sort_index(inplace=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    # This plot only has one main curve: Accuracy.
    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='o', linestyle='-', label='Accuracy with Filler Text')
    
    if show_nr:
        ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy (0% Filler) ({no_reasoning_accuracy:.2f}%)')
    if show_baseline:
        ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    restriction_str = " (Restricted)" if is_restricted else " (Full Dataset)"
    base_title = f'Accuracy vs. Percent of CoT Replaced by Filler ({model_name.upper()} on {dataset_name.upper()}){restriction_str}'
    subtitle = f'(Aggregated Across {num_questions} Questions)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Replaced by Filler Text', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Conditions', loc='best'); fig.tight_layout()

    # --- Output Path Construction ---
    output_plot_dir = os.path.join(plots_dir, model_name, 'filler_text', dataset_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"filler_text_{model_name}_{dataset_name}{suffix}"
    
    png_path = os.path.join(output_plot_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_plot_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()


def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_flags: dict):
    """
    Main function to orchestrate the filler text analysis for a single dataset.
    """
    print(f"\n--- Generating Filler Text Analysis for: {model_name.upper()} on {dataset_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name, is_restricted)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name, is_restricted)
        filler_df = load_results(model_name, results_dir, 'filler_text', dataset_name, is_restricted)
    except FileNotFoundError:
        return

    if filler_df.empty:
        print("  - No valid filler text data found. Skipping analysis.")
        return

    print("Generating main plot...")
    plot_single_graph(filler_df, baseline_df, no_reasoning_df, model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Filler Text plots for LALM results.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--restricted', action='store_true')
    parser.add_argument('--save-pdf', action='store_true')
    # "Opt-In" flags for controlling which elements are visible on the plot.
    # Note: This experiment only has benchmarks, not multiple curves.
    parser.add_argument('--show-baseline-benchmark', action='store_true')
    parser.add_argument('--show-nr-benchmark', action='store_true')
    
    # These three are irrelevant flags, but we include them for error handling for consistency across all the scripts.
    parser.add_argument('--grouped', action='store_true', help="Display the per-step plots.")
    parser.add_argument('--show-accuracy-curve', action='store_true', help="Display the accuracy curve (enabled by default for this plot).")
    parser.add_argument('--show-consistency-curve', action='store_true', help="Display the consistency curve (not applicable for this plot).")
    
    args = parser.parse_args()
    
    
    
    # This block handles the user input for the irrelevant flags, but we include error handling for consistency across all the scripts.
    if args.show_consistency_curve:
        # If the user explicitly asks for a consistency curve, we must inform them
        # that it's not available for this specific experiment and then continue w/o consistency metric.
        print("INFO: --show-consistency-curve is not applicable to the 'filler_text' experiment as consistency is not a measured metric for this experiment, only accuracy is calculated. So plots generation is continuing, but with no consistency line graph.")

    if args.show_accuracy_curve:
        # This plot's primary purpose is to show the accuracy curve. So it is shown by default irrespective of the flag.
        print("INFO: --show-accuracy-curve is the default and primary metric for this plot. It will always be shown.")
        
    if args.grouped:
        print("INFO: Grouped/per-step plots are not applicable for this experiment, it takes in a single chain, which is the longest chain for an individual sample.")
        
    
    
    
    # This experiment does not have accuracy/consistency curves to toggle.
    # We only need to pass the benchmark flags.
    show_flags = {
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
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.restricted, args.save_pdf, show_flags)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at {baseline_dir}.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.restricted, args.save_pdf, show_flags)
