# analysis/plot_filler_text.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, no_cot_df: pd.DataFrame, model_name: str, dataset_name: str, plots_dir: str, save_as_pdf: bool):
    """
    Generates and saves a single plot for the 'Filler Text' experiment.
    """
    # The number of chains is calculated from the unique questions in the filler_text df.
    # Note: This assumes one chain per question for this experiment's results.
    num_questions = len(df['id'].unique())
    
    # --- Context-Aware Benchmark Calculation (Question-Level) ---
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
    accuracy_curve = df.groupby('percentile')['is_correct'].mean() * 100
    accuracy_curve[0] = no_reasoning_accuracy
    accuracy_curve.sort_index(inplace=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    ax.plot(accuracy_curve.index, accuracy_curve.values, marker='o', linestyle='-', label='Accuracy with Filler Text')
    
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy (0% Filler) ({no_reasoning_accuracy:.2f}%)')
    if no_cot_accuracy is not None:
        ax.axhline(y=no_cot_accuracy, color='purple', linestyle=':', label=f'No-CoT Accuracy ({no_cot_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Accuracy vs. Percent of CoT Replaced by Filler ({model_name.upper()} on {dataset_name.upper()})'
    # Updated subtitle to reflect the data structure
    subtitle = f'(Aggregated Across {num_questions} Questions)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Replaced by Filler Text', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Conditions', loc='best'); fig.tight_layout()

    # --- Model-Agnostic Output Path ---
    output_plot_dir = os.path.join(plots_dir, model_name, 'filler_text', dataset_name)
    os.makedirs(output_plot_dir, exist_ok=True)
    base_filename = f"filler_text_{model_name}_{dataset_name}"
    
    png_path = os.path.join(output_plot_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_plot_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()


def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, include_no_cot: bool, save_as_pdf: bool):
    """ Main function to orchestrate the filler text analysis. """
    print(f"\n--- Generating Filler Text Analysis for: {model_name.upper()} on {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name)
        filler_df = load_results(model_name, results_dir, 'filler_text', dataset_name)
        no_cot_df = load_results(model_name, results_dir, 'no_cot', dataset_name) if include_no_cot else None
    except FileNotFoundError:
        return

    if filler_df.empty:
        print("  - No valid filler text data found. Skipping analysis.")
        return

    print("Generating main plot...")
    plot_single_graph(filler_df, baseline_df, no_reasoning_df, no_cot_df, model_name, dataset_name, plots_dir, save_as_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Filler Text plots for LALM results.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen').")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--include-no-cot', action='store_true')
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of each plot.")
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            exp_dir = os.path.join(args.results_dir, args.model, 'filler_text')
            dataset_names = sorted([f.replace(f'filler_text_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(exp_dir) if f.endswith('.jsonl')])
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.include_no_cot, args.save_pdf)
        except FileNotFoundError:
            print(f"Could not find filler_text directory for model '{args.model}' at {exp_dir}. Cannot run for 'all' datasets.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.include_no_cot, args.save_pdf)