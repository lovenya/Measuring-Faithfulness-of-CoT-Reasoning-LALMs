# analysis/final_plots/plot_delta_slope_vs_acc_early_answering.py

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

def calculate_delta_slope(x_coords, y_coords, experiment_name: str):
    """
    Calculates the end-to-end slope of a curve, ignoring artificial start/end points.
    """
    points = sorted(zip(x_coords, y_coords))

    # Filter out the artificial, hard-coded data points based on the experiment type.
    if experiment_name in ["early_answering", "adding_mistakes"]:
        # For these, the 100% point is the artificial baseline.
        filtered_points = [p for p in points if p[0] < 100]
    elif experiment_name in ["paraphrasing", "random_partial_filler_text"]:
        # For these, the 0% point is the artificial baseline.
        filtered_points = [p for p in points if p[0] > 0]
    else:
        filtered_points = points

    # If fewer than two points remain, a slope cannot be calculated.
    if len(filtered_points) < 2:
        return 0

    # Get the first and last points from the *remaining* data.
    x_first, y_first = filtered_points[0]
    x_last, y_last = filtered_points[-1]
    
    delta_y = y_last - y_first
    delta_x = x_last - x_first

    # Handle the edge case of a vertical line, though unlikely with our data.
    if delta_x == 0:
        return 0
        
    # Return the raw slope; the sign is meaningful.
    return delta_y / delta_x

def create_analysis(models: list, results_dir: str, plots_dir: str, acc_metric: str):
    """
    Generates a scatter plot of baseline accuracy vs. the Delta Slope of the
    consistency curve for the Early Answering experiment.
    """
    experiment_name = "early_answering"
    print(f"\n--- Generating Final Delta Slope vs. Accuracy Plot for: {experiment_name.upper()} ---")
    
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

                    # --- X-Axis: Delta Slope Calculation ---
                    exp_df = load_results(model_name, results_dir, experiment_name, dataset, is_restricted=True)
                    exp_df = exp_df[exp_df['total_sentences_in_chain'] > 0].copy()
                    if exp_df.empty:
                        print(f"    - WARNING: No valid data for '{dataset}'. Skipping.")
                        continue
                    
                    exp_df['percent_reasoning_provided'] = (exp_df['num_sentences_provided'] / exp_df['total_sentences_in_chain']) * 100
                    exp_df['percent_binned'] = (exp_df['percent_reasoning_provided'] / 5).round() * 5
                    
                    consistency_curve = exp_df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
                    
                    delta_slope = calculate_delta_slope(consistency_curve.index.tolist(), consistency_curve.values.tolist(), experiment_name)
                    
                    plot_data.append({
                        "model": model_name,
                        "dataset": dataset,
                        "baseline_accuracy": baseline_accuracy,
                        "delta_slope_score": delta_slope
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
            
            ax.scatter(data_point['delta_slope_score'], data_point['baseline_accuracy'],
                       color=color, marker=marker, s=200, label=f"{model_name} - {dataset_name}",
                       edgecolor='black', linewidth=0.5)

    ax.set_title(f'Baseline Accuracy vs. Reasoning Fragility (Early Answering)', fontsize=fontsize)
    ax.set_xlabel('Delta Slope (Fragility)', fontsize=fontsize)
    ax.set_ylabel(f'Baseline Accuracy ({acc_metric.capitalize()})', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=(fontsize-6), title="Model - Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # --- File Saving ---
    output_dir = os.path.join(plots_dir, "delta_slope_vs_accuracy")
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"delta_slope_vs_acc_{experiment_name}_{acc_metric}"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a scatter plot of baseline accuracy vs. delta slope score for the Early Answering experiment.")
    parser.add_argument('--models', nargs='+', required=True, help="A list of models to include in the plot (e.g., 'qwen' 'salmonn').")
    parser.add_argument('--acc-metric', type=str, default='macro', choices=['macro', 'chain'], help="The accuracy metric to use for the y-axis.")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./final_plots')
    args = parser.parse_args()
    
    create_analysis(args.models, args.results_dir, args.plots_dir, args.acc_metric)