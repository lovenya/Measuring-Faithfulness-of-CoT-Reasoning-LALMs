# analysis/final_plots/plot_final_early_answering.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

# Add the parent directory to the path to allow importing 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results

# --- Final Plot Style Guide ---
# This dictionary maps full dataset names to their final, publication-ready
# labels, colors, and marker styles.
FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "C0", "marker": "x"},
    "sakura-animal":   {"label": "S.Animal",   "color": "C1", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "C2", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "C3", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "C4", "marker": ">"}
}

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: bool, print_line_data: bool, save_stats: bool, save_pdf: bool):
    """
    Generates a final, cross-dataset consistency plot for the Early Answering experiment
    with the new, publication-quality aesthetic.
    """
    experiment_name = "early_answering"
    print(f"\n--- Generating Final Cross-Dataset Plot for: {experiment_name.upper()} ({model_name.upper()}) ---")
    
    # --- Data Loading and Preparation ---
    all_dfs = []
    try:
        baseline_dir = os.path.join(results_dir, model_name, 'baseline')
        dataset_names = sorted(list(set([f.replace(f'baseline_{model_name}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
        print(f"Found restricted datasets to process: {dataset_names}")

        for dataset in dataset_names:
            try:
                df = load_results(model_name, results_dir, experiment_name, dataset, is_restricted=True)
                df = df[df['total_sentences_in_chain'] > 0].copy()
                if not df.empty:
                    df['percent_reasoning_provided'] = (df['num_sentences_provided'] / df['total_sentences_in_chain']) * 100
                    df['dataset'] = dataset
                    all_dfs.append(df)
                else:
                    print(f"  - WARNING: No valid data for '{dataset}' in {experiment_name} results. Skipping.")
            except FileNotFoundError:
                print(f"  - WARNING: '{experiment_name}' results for dataset '{dataset}' not found. Skipping.")
                continue
        
        if not all_dfs:
            print("No data found for any dataset. Halting analysis.")
            return
            
        super_df = pd.concat(all_dfs, ignore_index=True)
        super_df['percent_binned'] = (super_df['percent_reasoning_provided'] / 5).round() * 5

    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}' at {baseline_dir}.")
        return

    # --- Prepare Output Path ---
    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"cross_dataset_{experiment_name}_{model_name}-restricted"
    
    # --- Statistical Analysis & Optional Output ---
    if print_line_data or save_stats:
        stats_output = []
        for dataset_name in sorted(super_df['dataset'].unique()):
            group_df = super_df[super_df['dataset'] == dataset_name]
            
            stats_output.append("="*60)
            stats_output.append(f"Dataset: {dataset_name}")
            stats_output.append("="*60)
            
            consistency_curve = group_df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
            stats_output.append("\nAggregated Line Data (Consistency %):")
            stats_output.append(f"  X Coords: {consistency_curve.index.tolist()}")
            stats_output.append(f"  Y Coords: {[round(y, 2) for y in consistency_curve.values.tolist()]}")

            stats_output.append("\nPer-Bin Distributional Stats (Per-Question Consistency %):")
            per_question_consistency = group_df.groupby(['id', 'percent_binned'])['is_consistent_with_baseline'].mean() * 100
            
            for bin_val in sorted(per_question_consistency.index.get_level_values('percent_binned').unique()):
                bin_stats = per_question_consistency.loc[:, bin_val].describe()
                stats_output.append(f"  - Bin {int(bin_val)}%:")
                stats_output.append(f"    - Mean:   {bin_stats.get('mean', 0):.2f}%")
                stats_output.append(f"    - Median: {bin_stats.get('50%', 0):.2f}%")
                stats_output.append(f"    - Std Dev: {bin_stats.get('std', 0):.2f}")
                stats_output.append(f"    - Min/Max: {bin_stats.get('min', 0):.2f}% / {bin_stats.get('max', 0):.2f}%")
                stats_output.append(f"    - IQR:    {bin_stats.get('25%', 0):.2f}% - {bin_stats.get('75%', 0):.2f}%")
            stats_output.append("\n")

        full_stats_string = "\n".join(stats_output)
        if print_line_data:
            print(full_stats_string)
        if save_stats:
            stats_path = os.path.join(output_dir, f"{base_filename}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write(full_stats_string)
            print(f"  - Statistical summary saved to: {stats_path}")

    # --- Plotting ---
    fontsize = 24
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    for dataset_name in sorted(super_df['dataset'].unique()):
        dataset_df = super_df[super_df['dataset'] == dataset_name]
        consistency_curve = dataset_df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
        
        style = FINAL_PLOT_STYLES.get(dataset_name, {"label": dataset_name, "color": "black", "marker": "."})
        
        ax.plot(consistency_curve.index, consistency_curve.values, 
                label=style['label'], 
                color=style['color'], 
                marker=style['marker'], 
                linestyle='-',
                linewidth=2,
                markersize=9)

    ax.set_title(f'Early Answering, {model_name.upper()}', fontsize=fontsize)
    ax.set_xlabel('Percentage % of Sentences Kept', fontsize=fontsize)
    ax.set_ylabel('Consistency (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    if y_zoom:
        ax.set_ylim(50, 100.5)
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)
    
    legend = ax.legend(
        title='Dataset', 
        fontsize=(fontsize - 4),  # Smaller font for legend text
        title_fontsize=(fontsize - 2), # Slightly larger font for legend title
        frameon=True, 
        facecolor='white', 
        framealpha=0.8
    )
    # --- END OF FIX ---

    ax.grid(True)
    fig.tight_layout()

    # --- File Saving ---
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final cross-dataset plots for the Early Answering experiment.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen', 'salmonn').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./final_plots')
    parser.add_argument('--y-zoom', action='store_true', help="Zoom the Y-axis to the 50-100% range.")
    parser.add_argument('--print-line-data', action='store_true', help="Print aggregated line data to the console.")
    parser.add_argument('--save-stats', action='store_true', help="Save a detailed statistical summary to a .txt file.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of the plot.")
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf)