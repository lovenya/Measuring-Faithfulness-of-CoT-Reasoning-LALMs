# analysis/cross_dataset/plot_final_adding_mistakes_0pct.py

"""
This script generates an alternative cross-dataset plot for the
'Adding Mistakes' experiment.

Unlike the standard plot which measures consistency against a separate
baseline run, this script measures consistency against the model's own
answer when NO mistakes have been introduced (the 0% state).

Since the Adding Mistakes experiment does not explicitly produce a "0 mistakes"
trial row, we use the `corresponding_baseline_predicted_choice` field which IS
the model's answer on the clean, unmodified CoT — i.e. the 0% state.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results, discover_datasets

FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "mmau":            {"label": "MMAU",       "color": "#a65628", "marker": "D"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"}
}

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: list, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool, perturbation_source: str = 'self'):
    experiment_name = "adding_mistakes"
    perturbation_label = f" [{perturbation_source.upper()}]" if perturbation_source != 'self' else ""
    print(f"\n--- Generating 0% Baseline Plot for: ADDING_MISTAKES{perturbation_label} ({model_name.upper()}) ---")
    
    try:
        dataset_names = discover_datasets(model_name, results_dir)
        print(f"Found datasets to process{perturbation_label}: {dataset_names}")
    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}'.")
        return

    all_dfs = []

    for dataset in dataset_names:
        try:
            df = load_results(model_name, results_dir, experiment_name, dataset, perturbation_source=perturbation_source)
        except FileNotFoundError:
            continue
            
        if df.empty:
            continue
            
        # Add basic info
        df['dataset'] = dataset
        
        # Calculate mistake position as percentage of chain
        df = df[df['total_sentences_in_chain'] > 0].copy()
        df['percent_before_mistake'] = ((df['mistake_position'] - 1) / df['total_sentences_in_chain']) * 100
        df['percent_binned'] = (df['percent_before_mistake'] / 5).round() * 5
        
        # In adding_mistakes, `is_consistent_with_baseline` already IS
        # consistency with the 0% (clean CoT) state, since the baseline
        # predicted choice is exactly the model's answer on clean CoT.
        # We just rename for clarity.
        df['is_consistent_with_0pct'] = df['is_consistent_with_baseline']
        
        all_dfs.append(df)

    if not all_dfs:
        print("No data found for any dataset. Halting analysis.")
        return
        
    super_df = pd.concat(all_dfs, ignore_index=True)

    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"cross_dataset_{experiment_name}_0pct_{model_name}"
    if perturbation_source != 'self':
        base_filename += f"-{perturbation_source}"
    
    # --- Convert to Percentage Scale for Plotting ---
    super_df['consistency_pct'] = super_df['is_consistent_with_0pct'].astype(int) * 100

    if print_line_data or save_stats:
        stats_output = []
        for dataset_name in sorted(super_df['dataset'].unique()):
            group_df = super_df[super_df['dataset'] == dataset_name]
            
            stats_output.append("="*60)
            stats_output.append(f"Dataset: {dataset_name} (Using 0% clean CoT as truth)")
            stats_output.append("="*60)
            
            consistency_curve = group_df.groupby('percent_binned')['is_consistent_with_0pct'].mean() * 100
            stats_output.append("\nAggregated Line Data (Consistency %):")
            stats_output.append(f"  X Coords: {consistency_curve.index.tolist()}")
            stats_output.append(f"  Y Coords: {[round(y, 2) for y in consistency_curve.values.tolist()]}")
            stats_output.append("\n")

        full_stats_string = "\n".join(stats_output)
        if print_line_data:
            print(full_stats_string)
        if save_stats:
            stats_path = os.path.join(output_dir, f"{base_filename}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write(full_stats_string)

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
    ax.set_title(f'Adding Mistakes (0% Ref){title_suffix}, {model_name.upper()}', fontsize=fontsize)
    ax.set_xlabel('Percentage % of Chain Without Mistake', fontsize=fontsize)
    ax.set_ylabel('Consistency with 0% Clean CoT (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)

    ax.grid(True)
    fig.tight_layout()

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 0% baseline cross-dataset plots for the Adding Mistakes experiment.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen_omni', 'flamingo_hf').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='plots/cross_dataset_plots')
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None, help="Set a custom Y-axis range.")
    parser.add_argument('--print-line-data', action='store_true', help="Print aggregated line data.")
    parser.add_argument('--save-stats', action='store_true', help="Save a detailed statistical summary.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of the plot.")
    parser.add_argument('--show-ci', action='store_true')
    parser.add_argument('--perturbation-source', type=str, default='self', choices=['self', 'mistral'])
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci, args.perturbation_source)
