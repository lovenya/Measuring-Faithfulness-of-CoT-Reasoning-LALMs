# analysis/cross_dataset/plot_final_adding_mistakes.py

"""
This script generates the final cross-dataset plot for the
'Adding Mistakes' experiment.

The scientific goal is to test faithfulness by inserting a mistake into a
reasoning chain and observing if the model's answer changes. This plot
visualizes consistency as a function of where the mistake was introduced.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import seaborn as sns

# Add the parent directory to the path to allow importing 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results, discover_datasets, check_completeness

# --- Final Plot Style Guide (Consistent across all scripts) ---
FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"}
}

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: list, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool, perturbation_source: str = 'self'):
    """
    Orchestrates the data loading, processing, and plotting for the Adding Mistakes experiment.
    """
    
    experiment_name = "adding_mistakes"
    perturbation_label = f" [{perturbation_source.upper()}]" if perturbation_source != 'self' else ""
    print(f"\n--- Generating Final Cross-Dataset Plot for: ADDING_MISTAKES{perturbation_label} ({model_name.upper()}) ---")
    
    # --- Dataset Discovery ---
    try:
        dataset_names = discover_datasets(model_name, results_dir)
        print(f"Found datasets to process{perturbation_label}: {dataset_names}")
    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}'.")
        return

    # --- Completeness Check & Data Loading ---
    all_dfs = []
    completeness_summary = []
    for dataset in dataset_names:
        # Load baseline
        try:
            baseline_df = load_results(model_name, results_dir, 'baseline', dataset)
        except FileNotFoundError:
            completeness_summary.append((dataset, "NO BASELINE", 0, 0, 0))
            continue
        
        # Load experiment results
        try:
            df = load_results(model_name, results_dir, experiment_name, dataset, perturbation_source=perturbation_source)
        except FileNotFoundError:
            completeness_summary.append((dataset, "NOT FOUND", len(set(zip(baseline_df['id'], baseline_df['chain_id']))), 0, 0))
            continue
        
        # Check completeness
        status = check_completeness(model_name, results_dir, experiment_name, dataset, baseline_df, df)
        label = "COMPLETE" if status['is_complete'] else "INCOMPLETE"
        completeness_summary.append((dataset, label, status['baseline_count'], status['experiment_count'], status['pct_complete']))
        
        # Process data for plotting
        df = df[df['total_sentences_in_chain'] > 0].copy()
        if not df.empty:
            df['percent_before_mistake'] = ((df['mistake_position'] - 1) / df['total_sentences_in_chain']) * 100
            df['dataset'] = dataset
            all_dfs.append(df)
    
    # --- Print Completeness Summary ---
    print(f"\n{'='*70}")
    print(f"  COMPLETENESS CHECK: {experiment_name.upper()}{perturbation_label} — {model_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20} {'Status':<12} {'Baseline':<10} {'Experiment':<12} {'Complete %':<10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for ds, label, bl, ex, pct in completeness_summary:
        print(f"  {ds:<20} {label:<12} {bl:<10} {ex:<12} {pct:>8.1f}%")
    print(f"{'='*70}\n")
    
    if not all_dfs:
        print("No data found for any dataset. Halting analysis.")
        return
        
    super_df = pd.concat(all_dfs, ignore_index=True)
    super_df['percent_binned'] = (super_df['percent_before_mistake'] / 10).round() * 10

    # --- Prepare Output Path ---
    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"cross_dataset_{experiment_name}_{model_name}"
    if perturbation_source != 'self':
        base_filename += f"-{perturbation_source}"
    
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
        
    title_suffix = " [Mistral]" if perturbation_source == 'mistral' else ""
    ax.set_title(f'Adding Mistakes{title_suffix}, {model_name.upper()}', fontsize=fontsize)
    ax.set_xlabel('Percentage % of Chain Without Mistake', fontsize=fontsize)
    ax.set_ylabel('Consistency (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)

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
    parser = argparse.ArgumentParser(description="Generate final cross-dataset plots for the Adding Mistakes experiment.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen_omni', 'flamingo_hf').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots/cross_dataset_plots')
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None, help="Set a custom Y-axis range (e.g., --y-zoom 45 100.5).")
    parser.add_argument('--print-line-data', action='store_true', help="Print aggregated line data to the console.")
    parser.add_argument('--save-stats', action='store_true', help="Save a detailed statistical summary to a .txt file.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of the plot.")
    parser.add_argument('--show-ci', action='store_true', help="Show the 95% confidence interval as a shaded region.")
    parser.add_argument('--perturbation-source', type=str, default='self', choices=['self', 'mistral'], help="Source of perturbations ('self' or 'mistral').")
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci, args.perturbation_source)