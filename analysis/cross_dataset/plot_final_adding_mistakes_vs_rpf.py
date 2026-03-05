# analysis/cross_dataset/plot_final_adding_mistakes_vs_rpf.py

"""
This script generates an alternative cross-dataset plot for the
'Adding Mistakes' experiment.

Unlike the standard plot which measures consistency against the true baseline,
this script measures consistency against the Random Partial Filler Text
experiment at 0% replacement. The idea is to control for the impact of prompt
formatting and experimental pipeline by comparing specifically against a
"zero modification" run of the same pipeline.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results, discover_datasets
from shared_plot_style import dataset_style, generate_dataset_legend, ordered_present_datasets

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: list, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool, perturbation_source: str = 'mistral', restricted: bool = False, rpf_filler_type: str = 'lorem'):
    experiment_name = "adding_mistakes"
    perturbation_label = f" [{perturbation_source.upper()}]" if perturbation_source != 'self' else ""
    print(f"\n--- Generating Plot for: ADDING_MISTAKES vs RPF@0%{perturbation_label} ({model_name.upper()}) ---")
    
    try:
        dataset_names = discover_datasets(model_name, results_dir, restricted=restricted)
        print(f"Found datasets to process{perturbation_label}: {dataset_names}")
    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}'.")
        return

    all_dfs = []
    missing_rpf0_report = []

    for dataset in dataset_names:
        # Load the RPF experiment (we need the 0% condition as truth)
        try:
            rpf_df = load_results(model_name, results_dir, 'random_partial_filler_text', dataset, filler_type=rpf_filler_type, restricted=restricted)
        except FileNotFoundError:
            missing_rpf0_report.append(f"[{dataset}] WARNING: Missing entire RPF experiment data (needed for ground truth). Skipping.")
            continue
            
        rpf_0 = rpf_df[rpf_df['percent_replaced'] == 0].copy()
        rpf_truth = dict(zip(zip(rpf_0['id'], rpf_0['chain_id']), rpf_0['predicted_choice']))
        
        if len(rpf_truth) == 0:
            missing_rpf0_report.append(f"[{dataset}] WARNING: RPF experiment found, but no 0% replacement condition samples found. Skipping.")
            continue

        # Load the Adding Mistakes experiment
        try:
            df = load_results(model_name, results_dir, experiment_name, dataset, perturbation_source=perturbation_source, restricted=restricted)
        except FileNotFoundError:
            continue
            
        if df.empty:
            continue
            
        # Add basic info
        df['dataset'] = dataset
        df = df[df['total_sentences_in_chain'] > 0].copy()
        df['percent_before_mistake'] = ((df['mistake_position'] - 1) / df['total_sentences_in_chain']) * 100
        df['percent_before_mistake_binned'] = (df['percent_before_mistake'] / 5).round() * 5

        # Only evaluate samples where we have an RPF 0% truth
        samples_with_truth = set(rpf_truth.keys())
        all_samples = set(zip(df['id'], df['chain_id']))
        missing_samples = all_samples - samples_with_truth
        
        if missing_samples:
            missing_rpf0_report.append(f"[{dataset}] WARNING: Dropping {len(missing_samples)} question(s) missing an RPF@0% run.")
            df = df[df.apply(lambda row: (row['id'], row['chain_id']) in samples_with_truth, axis=1)]
            
        # Calculate consistency against the RPF 0% truth
        def check_consistency(row):
            expected = rpf_truth.get((row['id'], row['chain_id']))
            return row['predicted_choice'] == expected

        df['is_consistent_with_rpf'] = df.apply(check_consistency, axis=1)

        # TASK 1: Add an explicit 100% endpoint by copying the RPF@0% anchor.
        # This gives a complete x-axis endpoint for adding_mistakes curves.
        # Since RPF@0% is the consistency truth itself, this endpoint is always 100%.
        chain_rows = (
            df[['id', 'chain_id', 'dataset']]
            .drop_duplicates()
            .copy()
        )
        if not chain_rows.empty:
            chain_rows['percent_before_mistake'] = 100.0
            chain_rows['percent_before_mistake_binned'] = 100.0
            chain_rows['is_consistent_with_rpf'] = True
            df = pd.concat([df, chain_rows], ignore_index=True, sort=False)

        # Filter out sparse bins (less than 10 samples) to remove extreme single-point outliers (e.g., MMAR)
        bin_counts = df['percent_before_mistake_binned'].value_counts()
        valid_bins = bin_counts[bin_counts >= 10].index
        df = df[df['percent_before_mistake_binned'].isin(valid_bins)]

        all_dfs.append(df)

    if missing_rpf0_report:
        print("\n" + "="*70)
        print("MISSING RPF@0% TRUTH REPORT")
        print("="*70)
        for line in missing_rpf0_report:
            print(line)
        print("="*70 + "\n")

    if not all_dfs:
        print("No valid data remaining for any dataset. Halting analysis.")
        return
        
    super_df = pd.concat(all_dfs, ignore_index=True)

    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"cross_dataset_{experiment_name}_vs_rpf_{model_name}"
    if restricted:
        base_filename += "_restricted"
    if perturbation_source != 'self':
        base_filename += f"-{perturbation_source}"
    
    # --- Convert to Percentage Scale for Plotting ---
    super_df['consistency_pct'] = super_df['is_consistent_with_rpf'].astype(int) * 100

    if print_line_data or save_stats:
        stats_output = []
        for dataset_name in sorted(super_df['dataset'].unique()):
            group_df = super_df[super_df['dataset'] == dataset_name]
            
            stats_output.append("="*60)
            stats_output.append(f"Dataset: {dataset_name} (Using RPF@0% as truth)")
            stats_output.append("="*60)
            
            consistency_curve = group_df.groupby('percent_before_mistake_binned')['is_consistent_with_rpf'].mean() * 100
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
            dataset_df.groupby('percent_before_mistake_binned')['is_consistent_with_rpf']
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

    title_suffix = f" [{perturbation_source.capitalize()}]" if perturbation_source != 'self' else ""
    restricted_label = " [Restricted]" if restricted else ""
    model_display = "Audio Flamingo 3" if model_name == "flamingo_hf" else "Qwen 2.5 Omni" if model_name == "qwen_omni" else model_name.upper()
    plt.title(f'Adding Mistakes, {model_display}')
    plt.xlabel('Percentage (%) of Chain Without Mistake')
    plt.ylabel('Consistency (%)')
    
    if y_zoom:
        plt.ylim(y_zoom[0], y_zoom[1])
    else:
        plt.ylim(-5, 105)
    plt.xlim(-5, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Paper style: keep figure clean (no in-plot legend).
    plt.tight_layout()

    # --- File Saving ---
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
    parser = argparse.ArgumentParser(description="Generate adding mistakes cross-dataset plots with RPF@0% as ground truth.")
    parser.add_argument('--model', type=str, required=True, help="Model name.")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='plots/cross_dataset_plots')
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None)
    parser.add_argument('--print-line-data', action='store_true')
    parser.add_argument('--save-stats', action='store_true')
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--show-ci', action='store_true')
    parser.add_argument('--perturbation-source', type=str, default='self', choices=['self', 'mistral'])
    parser.add_argument('--restricted', action='store_true')
    parser.add_argument('--rpf-filler-type', type=str, default='lorem', choices=['dots', 'lorem'])
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci, args.perturbation_source, args.restricted, args.rpf_filler_type)
