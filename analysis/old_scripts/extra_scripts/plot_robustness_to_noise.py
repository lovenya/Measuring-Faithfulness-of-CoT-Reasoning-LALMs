# analysis/plot_robustness_to_noise.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from .utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Generates and saves a single plot for a given group of 'Robustness to Noise' data.
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

    # --- Curve Generation ---
    # We group by the SNR level to get the average performance at each noise "dose".
    accuracy_curve = df.groupby('snr_db')['is_correct'].mean() * 100
    consistency_curve = df.groupby('snr_db')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy with Noisy Audio')
    ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Clean Audio Answer')

    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original (Clean Audio) CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Accuracy & Consistency vs. Audio Noise Level ({dataset_name.upper()})'
    subtitle = f'(Aggregated Across {num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('Signal-to-Noise Ratio (SNR) in dB (Higher is Cleaner)', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    
    # Reverse the x-axis so that cleaner audio is on the left.
    ax.invert_xaxis()
    
    ax.set_ylim(0, 105)
    ax.legend(title='Metrics', loc='best')
    fig.tight_layout()

    output_plot_dir = os.path.join(plots_dir, 'robustness_to_noise', dataset_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"robustness_to_noise_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


def create_analysis(dataset_name: str, results_dir: str, plots_dir: str, include_no_cot: bool):
    """ Main function to orchestrate the analysis. """
    print(f"\n--- Generating Robustness to Noise Analysis for: {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name)
        noise_df = load_results(results_dir, 'robustness_to_noise', dataset_name)
        no_cot_df = load_results(results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        print("  - Skipping plot due to missing one or more required result files.")
        return

    # --- Data Preparation ---
    # To plot consistency, we must add the 'is_consistent_with_baseline' column.
    # We merge with the baseline data to get the original predicted choice for each chain.
    baseline_predictions = baseline_df[['id', 'chain_id', 'predicted_choice']].rename(columns={'predicted_choice': 'baseline_predicted_choice'})
    combined_df = pd.merge(noise_df, baseline_predictions, on=['id', 'chain_id'], how='inner')
    combined_df['is_consistent_with_baseline'] = (combined_df['predicted_choice'] == combined_df['baseline_predicted_choice'])

    print("Generating main plot...")
    plot_single_graph(combined_df, baseline_df, no_reasoning_df, no_cot_df, 'aggregated', dataset_name, plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for the robustness to noise experiment.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to analyze ('mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--include-no-cot', action='store_true')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        baseline_dir = os.path.join(args.results_dir, 'baseline')
        dataset_names = sorted([f.replace('baseline_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])
        for dataset in dataset_names:
            create_analysis(dataset, args.results_dir, args.plots_dir, args.include_no_cot)
    else:
        create_analysis(args.dataset, args.results_dir, args.plots_dir, args.include_no_cot)