# analysis/cross_dataset/plot_final_early_answering_100pct.py

"""
This script generates an alternative cross-dataset plot for the
'Early Answering' experiment.

Unlike the standard plot which measures consistency against a separate
baseline run, this script measures consistency against the model's own
answer when 100% of the reasoning chain has been provided (the final step
within the early answering experiment itself).

This tells us: at what point does the model commit to its eventual final answer?
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results, discover_datasets
from shared_plot_style import dataset_style, generate_dataset_legend, ordered_present_datasets

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: list, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool, restricted: bool = False):
    experiment_name = "early_answering"
    print(f"\n--- Generating 100% Baseline Plot for: EARLY_ANSWERING ({model_name.upper()}) ---")
    
    try:
        dataset_names = discover_datasets(model_name, results_dir, restricted=restricted)
        print(f"Found datasets to process: {dataset_names}")
    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}'.")
        return

    all_dfs = []
    missing_100pct_report = []

    for dataset in dataset_names:
        try:
            df = load_results(model_name, results_dir, experiment_name, dataset, restricted=restricted)
        except FileNotFoundError:
            continue
            
        if df.empty:
            continue
            
        # Add basic info
        df['dataset'] = dataset
        df = df[df['total_sentences_in_chain'] > 0].copy()
        df['percent_reasoning_provided'] = (df['num_sentences_provided'] / df['total_sentences_in_chain']) * 100
        df['percent_binned'] = (df['percent_reasoning_provided'] / 5).round() * 5

        # 1. Isolate the 100% rows (max sentences provided per chain) to form our new "truth"
        max_per_chain = df.groupby(['id', 'chain_id'])['num_sentences_provided'].max().reset_index()
        max_per_chain.rename(columns={'num_sentences_provided': 'max_sentences'}, inplace=True)
        df = df.merge(max_per_chain, on=['id', 'chain_id'])
        
        full_cot_df = df[df['num_sentences_provided'] == df['max_sentences']]
        
        # Create a mapping dictionary: (id, chain_id) -> predicted_choice at 100%
        full_cot_truth = dict(zip(zip(full_cot_df['id'], full_cot_df['chain_id']), full_cot_df['predicted_choice']))
        
        # 2. Identify missing 100% samples
        all_samples = set(zip(df['id'], df['chain_id']))
        samples_with_100pct = set(full_cot_truth.keys())
        missing_samples = all_samples - samples_with_100pct
        
        if missing_samples:
            missing_100pct_report.append(f"\n[{dataset}] WARNING: Dropping {len(missing_samples)} question(s) missing a 100% run!")
            for q_id, chain_id in sorted(list(missing_samples)):
                missing_100pct_report.append(f"  -> Missing 100%: id={q_id}, chain_id={chain_id}")
            
            # Filter out the missing ones
            df = df[df.apply(lambda row: (row['id'], row['chain_id']) in samples_with_100pct, axis=1)]

        # 3. Calculate consistency against the 100% truth
        def check_consistency(row):
            expected = full_cot_truth.get((row['id'], row['chain_id']))
            return row['predicted_choice'] == expected

        df['is_consistent_with_100pct'] = df.apply(check_consistency, axis=1)
        
        # Drop the helper column
        df = df.drop(columns=['max_sentences'])
        
        all_dfs.append(df)

    if missing_100pct_report:
        print("\n" + "="*70)
        print("MISSING 100% RUN REPORT")
        print("="*70)
        for line in missing_100pct_report:
            print(line)
        print("="*70 + "\n")
    else:
        print("\nAll processed questions successfully found a 100% run. None dropped.")

    if not all_dfs:
        print("No valid data remaining for any dataset. Halting analysis.")
        return
        
    super_df = pd.concat(all_dfs, ignore_index=True)

    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"cross_dataset_{experiment_name}_100pct_{model_name}"
    if restricted:
        base_filename += "_restricted"
    
    # --- Convert to Percentage Scale for Plotting ---
    super_df['consistency_pct'] = super_df['is_consistent_with_100pct'].astype(int) * 100

    if print_line_data or save_stats:
        stats_output = []
        for dataset_name in sorted(super_df['dataset'].unique()):
            group_df = super_df[super_df['dataset'] == dataset_name]
            
            stats_output.append("="*60)
            stats_output.append(f"Dataset: {dataset_name} (Using 100% full CoT as truth)")
            stats_output.append("="*60)
            
            consistency_curve = group_df.groupby('percent_binned')['is_consistent_with_100pct'].mean() * 100
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
    if show_ci:
        print("INFO: --show-ci ignored for paper style parity (no CI bands).")

    plt.figure(figsize=(14, 10))
    for dataset_name in ordered_present_datasets(super_df['dataset'].unique()):
        dataset_df = super_df[super_df['dataset'] == dataset_name]
        style = dataset_style(dataset_name)

        consistency_curve = (
            dataset_df.groupby('percent_binned')['is_consistent_with_100pct']
            .mean()
            .mul(100)
            .sort_index()
        )
        plt.plot(
            consistency_curve.index.tolist(),
            consistency_curve.values.tolist(),
            color=style['color'],
            marker=style['marker'],
            linestyle='-',
        )

    restricted_label = " [Restricted]" if restricted else ""
    model_display = "Audio Flamingo 3" if model_name == "flamingo_hf" else "Qwen 2.5 Omni" if model_name == "qwen_omni" else model_name.upper()
    plt.title(f'Early Answering, {model_display}')
    plt.xlabel('Percentage (%) of the sentences kept')
    plt.ylabel('Consistency (%)')
    
    if y_zoom:
        plt.ylim(y_zoom[0], y_zoom[1])
    else:
        plt.ylim(-5, 105)
    plt.xlim(-5, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Paper style: no in-plot legend.
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  - Plot saved successfully to: {png_path}")

    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()

    legend_path = generate_dataset_legend(output_dir, super_df['dataset'].unique())
    print(f"  - Standalone dataset legend saved to: {legend_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 100% baseline cross-dataset plots for the Early Answering experiment.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to analyze (e.g., 'qwen_omni', 'flamingo_hf').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='plots/cross_dataset_plots')
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None, help="Set a custom Y-axis range.")
    parser.add_argument('--print-line-data', action='store_true', help="Print aggregated line data.")
    parser.add_argument('--save-stats', action='store_true', help="Save a detailed statistical summary.")
    parser.add_argument('--save-pdf', action='store_true', help="Save a PDF copy of the plot.")
    parser.add_argument('--show-ci', action='store_true')
    parser.add_argument('--restricted', action='store_true', help="Use restricted dataset (1-7 sentence CoTs).")
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci, args.restricted)
