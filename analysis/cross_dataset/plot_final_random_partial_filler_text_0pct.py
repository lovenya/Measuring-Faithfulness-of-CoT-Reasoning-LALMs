# analysis/cross_dataset/plot_final_random_partial_filler_text_0pct.py

"""
This script generates an alternative cross-dataset plot for the
'Random Partial Filler Text' experiment.

Unlike the standard plot which measures consistency against a separate
baseline run, this script measures consistency against the 0% replaced
trial from within the filler experiment itself.
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
    "mmau":            {"label": "MMAU",       "color": "#8c564b", "marker": "D"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"}
}

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: list, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool, filler_type: str = 'dots', restricted: bool = False):
    experiment_name = "random_partial_filler_text"
    filler_label = f" [{filler_type.upper()}]" if filler_type != 'dots' else ""
    print(f"\n--- Generating 0% Baseline Plot for: {experiment_name.upper()}{filler_label} ({model_name.upper()}) ---")
    
    try:
        dataset_names = discover_datasets(model_name, results_dir)
        print(f"Found datasets to process{filler_label}: {dataset_names}")
    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}'.")
        return

    all_dfs = []
    missing_0pct_report = []

    for dataset in dataset_names:
        try:
            df = load_results(model_name, results_dir, experiment_name, dataset, filler_type=filler_type, restricted=restricted)
        except FileNotFoundError:
            continue
            
        if df.empty:
            continue
            
        # Add basic info
        df['dataset'] = dataset
        df['percent_binned'] = (df['percent_replaced'] / 5).round() * 5

        # 1. Isolate the 0% rows to form our new "truth"
        zero_pct_df = df[df['percent_replaced'] == 0]
        
        # Create a mapping dictionary: (id, chain_id) -> predicted_choice
        zero_pct_truth = dict(zip(zip(zero_pct_df['id'], zero_pct_df['chain_id']), zero_pct_df['predicted_choice']))
        
        # 2. Identify missing 0% samples
        all_samples = set(zip(df['id'], df['chain_id']))
        samples_with_0pct = set(zero_pct_truth.keys())
        missing_samples = all_samples - samples_with_0pct
        
        if missing_samples:
            missing_0pct_report.append(f"\n[{dataset}] WARNING: Dropping {len(missing_samples)} question(s) missing a 0% replaced run!")
            for q_id, chain_id in sorted(list(missing_samples)):
                missing_0pct_report.append(f"  -> Missing 0%: id={q_id}, chain_id={chain_id}")
            
            # Filter out the missing ones
            df = df[df.apply(lambda row: (row['id'], row['chain_id']) in samples_with_0pct, axis=1)]

        # 3. Calculate consistency against the 0% truth
        def check_consistency(row):
            expected = zero_pct_truth.get((row['id'], row['chain_id']))
            return row['predicted_choice'] == expected

        df['is_consistent_with_0pct'] = df.apply(check_consistency, axis=1)
        
        # Add to main dataframe
        all_dfs.append(df)

    if missing_0pct_report:
        print("\n" + "="*70)
        print("MISSING 0% RUN REPORT")
        print("="*70)
        for line in missing_0pct_report:
            print(line)
        print("="*70 + "\n")
    else:
        print("\nAll processed questions successfully found a 0% replacement run. None dropped.")

    if not all_dfs:
        print("No valid data remaining for any dataset. Halting analysis.")
        return
        
    super_df = pd.concat(all_dfs, ignore_index=True)

    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    filler_suffix = f"_{filler_type}" if filler_type != 'dots' else ""
    base_filename = f"cross_dataset_{experiment_name}_0pct_{model_name}{filler_suffix}"
    if restricted:
        base_filename += "_restricted"
    
    # --- Convert to Percentage Scale for Plotting ---
    super_df['consistency_pct'] = super_df['is_consistent_with_0pct'].astype(int) * 100

    if print_line_data or save_stats:
        stats_output = []
        for dataset_name in sorted(super_df['dataset'].unique()):
            group_df = super_df[super_df['dataset'] == dataset_name]
            
            stats_output.append("="*60)
            stats_output.append(f"Dataset: {dataset_name} (Using 0% un-replaced CoT as truth)")
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
        
    title_prefix = "Random Partial Filler (0% Ref)"
    if filler_type == 'lorem':
        title_prefix += " (Lorem)"
    restricted_label = " [Restricted]" if restricted else ""
    ax.set_title(f'{title_prefix}{restricted_label}, {model_name.upper()}', fontsize=fontsize)
    ax.set_xlabel('Percentage % of Words Replaced', fontsize=fontsize)
    ax.set_ylabel('Consistency with 0% Replacement (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)

    ax.grid(True)
    
    # Add a small legend
    ax.legend(title='Dataset', title_fontsize=fontsize-14, fontsize=fontsize-18, loc='best')
    
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
    parser = argparse.ArgumentParser(description="Generate 0% baseline cross-dataset plots for the Random Partial Filler experiment.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen_omni', 'flamingo_hf').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='plots/cross_dataset_plots')
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None, help="Set a custom Y-axis range.")
    parser.add_argument('--print-line-data', action='store_true', help="Print aggregated line data.")
    parser.add_argument('--save-stats', action='store_true', help="Save a detailed statistical summary.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of the plot.")
    parser.add_argument('--show-ci', action='store_true')
    parser.add_argument('--filler-type', type=str, default='dots', choices=['dots', 'lorem'])
    parser.add_argument('--restricted', action='store_true', help="Use restricted dataset (1-7 sentence CoTs).")
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci, args.filler_type, args.restricted)

