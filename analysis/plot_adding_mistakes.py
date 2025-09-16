# analysis/plot_adding_mistakes.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, plot_group_name: str, model_name: str, dataset_name: str, plots_dir: str, save_as_pdf: bool):
    """
    Generates a single plot for 'Adding Mistakes' data.
    Uses chain-level benchmarks for grouped plots and conditional binning.
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
    
    no_cot_accuracy = None
    if no_cot_df is not None:
        relevant_no_cot_df = pd.merge(no_cot_df, df[['id']].drop_duplicates(), on='id')
        if not relevant_no_cot_df.empty:
            no_cot_accuracy = relevant_no_cot_df.groupby('id')['is_correct'].mean().mean() * 100

    # --- Curve Generation with Conditional Binning ---
    if plot_group_name == 'aggregated':
        df['percent_binned'] = (df['percent_before_mistake'] / 10).round() * 10
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        accuracy_curve = df.groupby('percent_before_mistake')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_before_mistake')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy After Mistake')
    ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Original Answer')

    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Accuracy & Consistency vs. Position of Introduced Mistake ({model_name.upper()} on {dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Before Mistake', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Model-Agnostic Output Path ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, model_name, 'adding_mistakes', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, model_name, 'adding_mistakes', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    
    base_filename = f"adding_mistakes_{model_name}_{dataset_name}_{plot_group_name}"
    
    # --- UPDATED SAVE LOGIC ---
    # 1. Save the standard PNG version
    png_path = os.path.join(output_plot_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    # 2. Conditionally save the PDF version
    if save_as_pdf:
        pdf_path = os.path.join(output_plot_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()
       
    
def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool, include_no_cot: bool, num_samples: int, num_chains: int,save_as_pdf: bool):
    """ Main function to orchestrate the 'Adding Mistakes' analysis. """
    print(f"\n--- Generating Adding Mistakes Analysis for: {model_name.upper()} on {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name)
        mistakes_df = load_results(model_name, results_dir, 'adding_mistakes', dataset_name)
        no_cot_df = load_results(model_name, results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        return

    # --- Data Filtering based on CLI arguments ---
    if num_samples is not None and num_samples > 0:
        unique_ids = baseline_df['id'].unique()[:num_samples]
        baseline_df = baseline_df[baseline_df['id'].isin(unique_ids)]
        no_reasoning_df = no_reasoning_df[no_reasoning_df['id'].isin(unique_ids)]
        mistakes_df = mistakes_df[mistakes_df['id'].isin(unique_ids)]
        if no_cot_df is not None:
            no_cot_df = no_cot_df[no_cot_df['id'].isin(unique_ids)]

    if num_chains is not None and num_chains > 0:
        baseline_df = baseline_df[baseline_df['chain_id'] < num_chains]
        mistakes_df = mistakes_df[mistakes_df['chain_id'] < num_chains]
        if no_cot_df is not None:
            no_cot_df = no_cot_df[no_cot_df['chain_id'] < num_chains]

    # --- "Meaningful Manipulation" Filter ---
    mistakes_df = mistakes_df[mistakes_df['total_sentences_in_chain'] > 0].copy()
    if mistakes_df.empty:
        print("  - No valid data with non-empty CoTs found. Skipping analysis.")
        return

    # Calculate the percentage of the CoT that precedes the introduced mistake.
    mistakes_df['percent_before_mistake'] = ((mistakes_df['mistake_position'] - 1) / mistakes_df['total_sentences_in_chain']) * 100

    print("Generating main aggregated plot...")
    plot_single_graph(mistakes_df, baseline_df, no_reasoning_df, no_cot_df, 'aggregated', model_name, dataset_name, plots_dir, save_as_pdf)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = mistakes_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, no_cot_df, f'{total_steps}_sentences', model_name, dataset_name, plots_dir, save_as_pdf)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data (<=10 chains).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Adding Mistakes plots for LALM results.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen').")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--include-no-cot', action='store_true')
    parser.add_argument('--num-samples', type=int, default=None, help="Limit analysis to the first N unique samples.")
    parser.add_argument('--num-chains', type=int, default=None, help="Limit analysis to the first N chains per sample.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of each plot in addition to the PNG.")
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            dataset_names = sorted([f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot, args.num_samples, args.num_chains, args.save_pdf)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot, args.num_samples, args.num_chains, args.save_pdf)