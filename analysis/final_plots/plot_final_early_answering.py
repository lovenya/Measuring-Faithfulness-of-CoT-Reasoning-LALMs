# analysis/final_plots/plot_final_early_answering.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import seaborn as sns

# Add the parent directory to the path to allow importing 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results

# --- Final Plot Style Guide ---
FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"}
}

# --- Hard-coded list of datasets ---
# This removes the dependency on the baseline directory for dataset discovery.
DATASET_NAMES = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: bool, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool):
    """
    Generates a final, self-contained, cross-dataset consistency plot for the Early Answering experiment.
    """
    experiment_name = "early_answering"
    print(f"\n--- Generating Final Cross-Dataset Plot for: {experiment_name.upper()} ({model_name.upper()}) ---")
    
    # --- Data Loading and Preparation ---
    all_dfs = []
    print(f"Processing hard-coded datasets: {DATASET_NAMES}")

    for dataset in DATASET_NAMES:
        try:
            # The script now only loads its own primary result files.
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

    # --- Convert to Percentage Scale for Plotting ---
    super_df['consistency_pct'] = super_df['is_consistent_with_baseline'].astype(int) * 100

    # --- Plotting ---
    fontsize = 24
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    for dataset_name, style in FINAL_PLOT_STYLES.items():
        if dataset_name not in super_df['dataset'].unique():
            continue
            
        dataset_df = super_df[super_df['dataset'] == dataset_name]
        
        sns.lineplot(data=dataset_df, 
                     x='percent_binned', 
                     y='consistency_pct',
                     label=style['label'], 
                     color=style['color'], 
                     marker=style['marker'], 
                     linestyle='-',
                     linewidth=2,
                     markersize=20,
                     ci=95 if show_ci else None,
                     ax=ax)
        
    ax.set_title(f'Early Answering, {model_name.upper()}', fontsize=fontsize)
    ax.set_xlabel('Percentage % of Sentences Kept', fontsize=fontsize)
    ax.set_ylabel('Consistency (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    if y_zoom:
        ax.set_ylim(25, 100.5)
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)
    
    legend = ax.legend(
        title='Dataset', 
        fontsize=(fontsize - 4),
        title_fontsize=(fontsize - 2),
        frameon=True, 
        facecolor='white', 
        framealpha=0.8
    )

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
    parser.add_argument('--y-zoom', action='store_true', help="Zoom the Y-axis to the 45-100% range.")
    parser.add_argument('--print-line-data', action='store_true', help="Print aggregated line data to the console.")
    parser.add_argument('--save-stats', action='store_true', help="Save a detailed statistical summary to a .txt file.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of the plot.")
    parser.add_argument('--show-ci', action='store_true', help="Show the 95% confidence interval as a shaded region.")
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci)