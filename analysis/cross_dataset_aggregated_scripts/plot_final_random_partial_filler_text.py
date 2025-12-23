# analysis/cross_dataset_aggregated_scripts/plot_final_random_partial_filler_text.py

"""
This script generates the final, publication-quality cross-dataset plot for the
'Random Partial Filler Text' experiment.

The scientific goal is to test the importance of the reasoning content by
replacing a random percentage of words in the CoT with meaningless filler.
This plot visualizes the model's consistency as a function of how much of the
reasoning chain has been randomly corrupted.

This script is hard-coded to run on the 'restricted' data subset (1-6 step CoTs)
and produces a single, cross-dataset aggregated plot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import seaborn as sns

# Add the parent directory to the path to allow importing 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results

# --- Final Plot Style Guide (Consistent across all scripts) ---
FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"}
}

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: list, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool, filler_type: str = 'dots'):
    """
    Orchestrates the data loading, processing, and plotting for the Random Partial Filler experiment.

    Args:
        model_name (str): The name of the model to analyze.
        results_dir (str): The root directory for the results.
        plots_dir (str): The root directory where plots will be saved.
        y_zoom (list | None): A list of two floats for the y-axis range, or None.
        print_line_data (bool): Flag to print aggregated line data to the console.
        save_stats (bool): Flag to save a detailed statistical summary to a file.
        save_pdf (bool): Flag to save a PDF copy of the plot.
        show_ci (bool): Flag to show the 95% confidence interval on the plot.
        filler_type (str): The type of filler used ('dots' or 'lorem').
    """
    
    experiment_name = "random_partial_filler_text"
    print(f"\n--- Generating Final Cross-Dataset Plot for: {experiment_name.upper()} ({model_name.upper()}) ---")
    
    # --- Data Loading and Preparation ---
    all_dfs = []
    try:
        baseline_dir = os.path.join(results_dir, model_name, 'baseline')
        dataset_names = sorted(list(set([f.replace(f'baseline_{model_name}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
        print(f"Found restricted datasets to process: {dataset_names}")

        for dataset in dataset_names:
            try:
                # We need baseline for consistency check and early_answering for sentence counts
                baseline_df = load_results(model_name, results_dir, 'baseline', dataset, is_restricted=True)
                early_df = load_results(model_name, results_dir, 'early_answering', dataset, is_restricted=True)
                df = load_results(model_name, results_dir, experiment_name, dataset, is_restricted=True, filler_type=filler_type)

                # Add consistency and sentence count metadata
                baseline_predictions = baseline_df[['id', 'chain_id', 'predicted_choice']].rename(columns={'predicted_choice': 'baseline_predicted_choice'})
                df = pd.merge(df, baseline_predictions, on=['id', 'chain_id'], how='inner')
                df['is_consistent_with_baseline'] = (df['predicted_choice'] == df['baseline_predicted_choice'])
                
                sentence_counts = early_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
                df = pd.merge(df, sentence_counts, on=['id', 'chain_id'], how='inner')

                df = df[df['total_sentences_in_chain'] > 0].copy()
                if not df.empty:
                    df['dataset'] = dataset
                    all_dfs.append(df)
                else:
                    print(f"  - WARNING: No valid data for '{dataset}' in {experiment_name} results. Skipping.")
            except FileNotFoundError:
                print(f"  - WARNING: '{experiment_name}' or dependent results for dataset '{dataset}' not found. Skipping.")
                continue
        
        if not all_dfs:
            print("No data found for any dataset. Halting analysis.")
            return
            
        super_df = pd.concat(all_dfs, ignore_index=True)
        super_df['percent_binned'] = (super_df['percent_replaced'] / 5).round() * 5

    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}' at {baseline_dir}.")
        return

    # --- Hard-code 0% to be 100% consistent ---
    # We locate all rows where the binned percentage is 0 and set their consistency to True.
    # This ensures the plot and stats correctly reflect the theoretical baseline.
    super_df.loc[super_df['percent_binned'] == 0, 'is_consistent_with_baseline'] = True

    # --- Prepare Output Path ---
    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    filler_suffix = f"_{filler_type}" if filler_type != 'dots' else ""
    base_filename = f"cross_dataset_{experiment_name}_{model_name}-restricted{filler_suffix}"
    
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
    fontsize = 32
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
                     errorbar=('ci', 95) if show_ci else None,
                     ax=ax,
                     legend=False)
        
    # Update plot titles and labels for this specific experiment
    title_prefix = "Random Partial Filler"
    if filler_type == 'lorem':
        title_prefix += " (Lorem Ipsum)"
    ax.set_title(f'{title_prefix}, {model_name.upper()}', fontsize=fontsize)
    ax.set_xlabel('Percentage % of Words Replaced', fontsize=fontsize)
    ax.set_ylabel('Consistency (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    
    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)
    
    # legend = ax.legend(
    #     title='Dataset', 
    #     fontsize=(fontsize - 4),
    #     title_fontsize=(fontsize - 2),
    #     frameon=True, 
    #     facecolor='white', 
    #     framealpha=0.8
    # )

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
    parser = argparse.ArgumentParser(description="Generate final cross-dataset plots for the Random Partial Filler experiment.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen', 'salmonn').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='plots/cross_dataset_plots', help="The root directory for final plots.")
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None, help="Set a custom Y-axis range (e.g., --y-zoom 45 100.5).")
    parser.add_argument('--print-line-data', action='store_true', help="Print aggregated line data to the console.")
    parser.add_argument('--save-stats', action='store_true', help="Save a detailed statistical summary to a .txt file.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of the plot.")
    parser.add_argument('--show-ci', action='store_true', help="Show the 95% confidence interval as a shaded region.")
    parser.add_argument('--filler-type', type=str, default='dots', choices=['dots', 'lorem'], help="Type of filler to analyze.")
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci, args.filler_type)
