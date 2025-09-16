# analysis/plot_early_answering.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, plot_group_name: str, model_name: str, dataset_name: str, plots_dir: str):
    """
    Generates and saves a single plot for a given group of 'Early Answering' data.
    This function correctly applies binning ONLY to the 'aggregated' plot.
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
        df['percent_binned'] = (df['percent_reasoning_provided'] / 5).round() * 5
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        accuracy_curve = df.groupby('percent_reasoning_provided')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_reasoning_provided')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy at Step')
    ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Final Answer')

    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Final CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Accuracy & Consistency vs. Reasoning Progression ({model_name.upper()} on {dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Provided', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Model-Agnostic Output Path ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, model_name, 'early_answering', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, model_name, 'early_answering', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"early_answering_{model_name}_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool, include_no_cot: bool, num_samples: int, num_chains: int):
    """ Main function to orchestrate the early answering analysis. """
    print(f"\n--- Generating Early Answering Analysis for: {model_name.upper()} on {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name)
        early_df = load_results(model_name, results_dir, 'early_answering', dataset_name)
        no_cot_df = load_results(model_name, results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        return

    # --- Data Filtering based on CLI arguments ---
    if num_samples is not None and num_samples > 0:
        unique_ids = baseline_df['id'].unique()[:num_samples]
        baseline_df = baseline_df[baseline_df['id'].isin(unique_ids)]
        no_reasoning_df = no_reasoning_df[no_reasoning_df['id'].isin(unique_ids)]
        early_df = early_df[early_df['id'].isin(unique_ids)]
        if no_cot_df is not None:
            no_cot_df = no_cot_df[no_cot_df['id'].isin(unique_ids)]

    if num_chains is not None and num_chains > 0:
        baseline_df = baseline_df[baseline_df['chain_id'] < num_chains]
        early_df = early_df[early_df['chain_id'] < num_chains]
        if no_cot_df is not None:
            no_cot_df = no_cot_df[no_cot_df['chain_id'] < num_chains]
    
    # --- "Meaningful Manipulation" Filter ---
    # This experiment is only valid for chains that have at least one sentence.
    early_df = early_df[early_df['total_sentences_in_chain'] > 0].copy()
    if early_df.empty:
        print("  - No valid data with non-empty CoTs found. Skipping analysis.")
        return

    # Calculate the percentage of the CoT provided at each step.
    early_df['percent_reasoning_provided'] = (early_df['num_sentences_provided'] / early_df['total_sentences_in_chain']) * 100

    print("Generating main aggregated plot...")
    plot_single_graph(early_df, baseline_df, no_reasoning_df, no_cot_df, 'aggregated', model_name, dataset_name, plots_dir)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = early_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, no_cot_df, f'{total_steps}_sentences', model_name, dataset_name, plots_dir)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data (<=10 chains).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Early Answering plots for LALM results.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen').")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--include-no-cot', action='store_true')
    parser.add_argument('--num-samples', type=int, default=None, help="Limit analysis to the first N unique samples.")
    parser.add_argument('--num-chains', type=int, default=None, help="Limit analysis to the first N chains per sample.")
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            dataset_names = sorted([f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot, args.num_samples, args.num_chains)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at {baseline_dir}. Cannot run for 'all' datasets.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.grouped, args.include_no_cot, args.num_samples, args.num_chains)