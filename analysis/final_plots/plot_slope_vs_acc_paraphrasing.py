# analysis/final_plots/plot_slope_vs_acc_paraphrasing.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np

# Add the parent directory to the path to allow importing 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results

# --- Final Plot Style Guides ---
DATASET_COLORS = {
    "mmar": "#e41a1c",
    "sakura-animal": "#377eb8",
    "sakura-emotion": "#4daf4a",
    "sakura-gender": "#ff7f00",
    "sakura-language": "#984ea3"
}
MODEL_MARKERS = {
    "qwen": "o",
    "salmonn": "s",
    "flamingo": "X"
}

def calculate_mean_slope(x_coords, y_coords):
    """Calculates the mean of the slopes between adjacent points."""
    slopes = []
    for i in range(len(x_coords) - 1):
        delta_y = y_coords[i+1] - y_coords[i]
        delta_x = x_coords[i+1] - x_coords[i]
        if delta_x != 0:
            slopes.append(delta_y / delta_x)
    
    if not slopes:
        return 0
    return abs(np.mean(slopes))

def create_analysis(models: list, results_dir: str, plots_dir: str, acc_metric: str):
    """
    Generates a final scatter plot correlating baseline accuracy with the mean slope
    of the consistency curve for the Paraphrasing experiment.
    """
    experiment_name = "paraphrasing"
    print(f"\n--- Generating Final Slope vs. Accuracy Plot for: {experiment_name.upper()} ---")
    
    plot_data = []

    for model_name in models:
        print(f"\nProcessing model: {model_name.upper()}")
        try:
            baseline_dir = os.path.join(results_dir, model_name, 'baseline')
            dataset_names = sorted(list(set([f.replace(f'baseline_{model_name}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
            
            for dataset in dataset_names:
                print(f"  - Processing dataset: {dataset}")
                try:
                    # --- Y-Axis: Baseline Accuracy Calculation ---
                    baseline_df = load_results(model_name, results_dir, 'baseline', dataset, is_restricted=True)
                    if acc_metric == 'macro':
                        baseline_accuracy = baseline_df.groupby('id')['is_correct'].mean().mean() * 100
                    else: # chain-level
                        baseline_accuracy = baseline_df['is_correct'].mean() * 100

                    # --- X-Axis: Mean Slope Calculation ---
                    exp_df = load_results(model_name, results_dir, experiment_name, dataset, is_restricted=True)
                    exp_df = exp_df[exp_df['total_sentences_in_chain'] > 0].copy()
                    if exp_df.empty:
                        print(f"    - WARNING: No valid data for '{dataset}'. Skipping.")
                        continue
                    
                    exp_df['percent_paraphrased'] = (exp_df['num_sentences_paraphrased'] / exp_df['total_sentences_in_chain']) * 100
                    exp_df['percent_binned'] = (exp_df['percent_paraphrased'] / 10).round() * 10
                    
                    # Synthetically add the 0% data point for a complete curve
                    zero_percent_df = exp_df.drop_duplicates(subset=['id', 'chain_id']).copy()
                    zero_percent_df['percent_binned'] = 0
                    zero_percent_df['is_consistent_with_baseline'] = True
                    exp_df = pd.concat([exp_df, zero_percent_df], ignore_index=True)

                    consistency_curve = exp_df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
                    
                    mean_slope = calculate_mean_slope(consistency_curve.index.tolist(), consistency_curve.values.tolist())
                    
                    plot_data.append({
                        "model": model_name,
                        "dataset": dataset,
                        "baseline_accuracy": baseline_accuracy,
                        "mean_slope_score": mean_slope
                    })

                except FileNotFoundError:
                    print(f"    - WARNING: Results for '{dataset}' not found. Skipping.")
                    continue
        except FileNotFoundError:
            print(f"  - WARNING: Baseline directory for model '{model_name}' not found. Skipping model.")
            continue
            
    if not plot_data:
        print("\nNo data available to plot. Halting analysis.")
        return

    plot_df = pd.DataFrame(plot_data)
    print("\n--- Generated Plot Data ---")
    print(plot_df)
    print("-------------------------\n")

    # --- Plotting ---
    fontsize = 20
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

    for model_name in plot_df['model'].unique():
        model_df = plot_df[plot_df['model'] == model_name]
        marker = MODEL_MARKERS.get(model_name, 'P')
        
        for dataset_name in model_df['dataset'].unique():
            data_point = model_df[model_df['dataset'] == dataset_name]
            color = DATASET_COLORS.get(dataset_name, 'black')
            
            ax.scatter(data_point['mean_slope_score'], data_point['baseline_accuracy'],
                       color=color, marker=marker, s=200, label=f"{model_name} - {dataset_name}",
                       edgecolor='black', linewidth=0.5)

    ax.set_title(f'Baseline Accuracy vs. Reasoning Fragility (Paraphrasing)', fontsize=fontsize)
    ax.set_xlabel('Mean Slope Score (Fragility)', fontsize=fontsize)
    ax.set_ylabel(f'Baseline Accuracy ({acc_metric.capitalize()})', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=(fontsize-6), title="Model - Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # --- File Saving ---
    output_dir = os.path.join(plots_dir, "slope_vs_accuracy")
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"slope_vs_acc_{experiment_name}_{acc_metric}"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a scatter plot of baseline accuracy vs. mean slope score for the Paraphrasing experiment.")
    parser.add_argument('--models', nargs='+', required=True, help="A list of models to include in the plot (e.g., 'qwen' 'salmonn').")
    parser.add_argument('--acc-metric', type=str, default='macro', choices=['macro', 'chain'], help="The accuracy metric to use for the y-axis.")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./final_plots')
    args = parser.parse_args()
    
    create_analysis(args.models, args.results_dir, args.plots_dir, args.acc_metric)