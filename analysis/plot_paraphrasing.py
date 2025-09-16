# analysis/plot_paraphrasing.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, plot_group_name: str, model_name: str, dataset_name: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_accuracy: bool, show_consistency: bool, show_baseline: bool, show_nr: bool):
    """
    Generates a single plot for 'Paraphrasing' data, with full feature control.
    """
    num_chains = len(df[['id', 'chain_id']].drop_duplicates())

    # --- Benchmark Calculation ---
    if plot_group_name == 'aggregated':
        relevant_ids = df[['id']].drop_duplicates()
        relevant_baseline_df = pd.merge(baseline_df, relevant_ids, on='id')
        relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_ids, on='id')
    else:
        relevant_ids = df[['id', 'chain_id']].drop_duplicates()
        relevant_baseline_df = pd.merge(baseline_df, relevant_ids, on=['id', 'chain_id'])
        relevant_no_reasoning_df = pd.merge(no_reasoning_df, df[['id']].drop_duplicates(), on='id')

    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100
    
    # --- Curve Generation with Conditional Binning ---
    if plot_group_name == 'aggregated':
        df['percent_binned'] = (df['percent_paraphrased'] / 10).round() * 10
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        accuracy_curve = df.groupby('percent_paraphrased')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_paraphrased')['is_consistent_with_baseline'].mean() * 100

    # Synthetically add the 0% data point.
    accuracy_curve[0] = baseline_accuracy
    consistency_curve[0] = 100.0
    accuracy_curve.sort_index(inplace=True)
    consistency_curve.sort_index(inplace=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    if show_accuracy:
        ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy with Paraphrased CoT')
    if show_consistency:
        ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Original Answer')
    if show_nr:
        ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if show_baseline:
        ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    restriction_str = " (Restricted)" if is_restricted else " (Full Dataset)"
    base_title = f'Accuracy & Consistency vs. Paraphrasing Progression ({model_name.upper()} on {dataset_name.upper()}){restriction_str}'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Paraphrased', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Output Path ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, model_name, 'paraphrasing', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, model_name, 'paraphrasing', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"paraphrasing_{model_name}_{dataset_name}_{plot_group_name}{suffix}"
    
    png_path = os.path.join(output_plot_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_plot_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()

def plot_cross_dataset_graph(df: pd.DataFrame, model_name: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_accuracy: bool, show_consistency: bool):
    """
    Generates a single plot comparing aggregated results across multiple datasets.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    palette = sns.color_palette("viridis", n_colors=df['dataset'].nunique())
    df['percent_binned'] = (df['percent_paraphrased'] / 10).round() * 10

    # We need to handle the 0% point for each dataset individually before plotting
    all_curves_df = []
    for dataset_name, group_df in df.groupby('dataset'):
        # Calculate baseline accuracy for this specific dataset to get the 0% point
        baseline_df = load_results(model_name, './results', 'baseline', dataset_name, is_restricted)
        baseline_accuracy = baseline_df.groupby('id')['is_correct'].mean().mean() * 100

        if show_accuracy:
            accuracy_curve = group_df.groupby('percent_binned')['is_correct'].mean().reset_index()
            accuracy_curve['is_correct'] *= 100
            accuracy_curve.loc[len(accuracy_curve)] = [0, baseline_accuracy] # Add 0% point
            accuracy_curve.sort_values('percent_binned', inplace=True)
            accuracy_curve['dataset'] = dataset_name
            accuracy_curve['metric'] = 'Accuracy'
            accuracy_curve.rename(columns={'is_correct': 'value'}, inplace=True)
            all_curves_df.append(accuracy_curve)

        if show_consistency:
            consistency_curve = group_df.groupby('percent_binned')['is_consistent_with_baseline'].mean().reset_index()
            consistency_curve['is_consistent_with_baseline'] *= 100
            consistency_curve.loc[len(consistency_curve)] = [0, 100.0] # Add 0% point
            consistency_curve.sort_values('percent_binned', inplace=True)
            consistency_curve['dataset'] = dataset_name
            consistency_curve['metric'] = 'Consistency'
            consistency_curve.rename(columns={'is_consistent_with_baseline': 'value'}, inplace=True)
            all_curves_df.append(consistency_curve)
    
    if not all_curves_df:
        print("  - No data to plot for cross-dataset comparison.")
        return

    plot_df = pd.concat(all_curves_df)
    
    sns.lineplot(data=plot_df, x='percent_binned', y='value', hue='dataset', style='metric', ax=ax, marker='o', palette=palette)

    plotted_metrics = plot_df['metric'].unique()
    restriction_str = " (Restricted)" if is_restricted else " (Full Dataset)"
    ax.set_title(f'Cross-Dataset Comparison of { " & ".join(plotted_metrics) }: Paraphrasing ({model_name.upper()}){restriction_str}', fontsize=16, pad=20)
    
    ax.set_xlabel('% of Reasoning Chain Paraphrased', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Legend'); fig.tight_layout()

    output_plot_dir = os.path.join(plots_dir, model_name, 'paraphrasing', 'cross_dataset')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"cross_dataset_paraphrasing_{model_name}{suffix}"
    
    png_path = os.path.join(output_plot_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Cross-dataset plot saved to: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_plot_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()

def create_cross_dataset_analysis(model_name: str, results_dir: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_flags: dict):
    """
    Loads data from ALL datasets and generates a single comparative plot.
    """
    print(f"\n--- Generating CROSS-DATASET Paraphrasing Analysis for: {model_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    all_dfs = []
    try:
        baseline_dir = os.path.join(results_dir, model_name, 'baseline')
        if is_restricted:
            dataset_names = sorted(list(set([f.replace(f'baseline_{model_name}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
        else:
            dataset_names = sorted(list(set([f.replace(f'baseline_{model_name}_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if not f.endswith('-restricted.jsonl')])))
        
        print(f"Found datasets to process: {dataset_names}")

        for dataset in dataset_names:
            try:
                df = load_results(model_name, results_dir, 'paraphrasing', dataset, is_restricted)
                df = df[df['total_sentences_in_chain'] > 0].copy()
                if not df.empty:
                    df['percent_paraphrased'] = (df['num_sentences_paraphrased'] / df['total_sentences_in_chain']) * 100
                    df['dataset'] = dataset
                    all_dfs.append(df)
                else:
                    print(f"  - WARNING: No valid data for '{dataset}' in paraphrasing results. Skipping.")
            except FileNotFoundError:
                print(f"  - WARNING: 'paraphrasing' results for dataset '{dataset}' not found. Skipping.")
                continue
        
        if not all_dfs:
            print("No data found for any dataset. Halting analysis.")
            return
            
        super_df = pd.concat(all_dfs, ignore_index=True)
        plot_cross_dataset_graph(super_df, model_name, plots_dir, is_restricted, save_as_pdf, **show_flags)

    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}' at {baseline_dir}.")
        return

def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, is_restricted: bool, generate_grouped: bool, save_as_pdf: bool, show_flags: dict):
    """ Main function to orchestrate the paraphrasing analysis. """
    print(f"\n--- Generating Paraphrasing Analysis for: {model_name.upper()} on {dataset_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name, is_restricted)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name, is_restricted)
        paraphrasing_df = load_results(model_name, results_dir, 'paraphrasing', dataset_name, is_restricted)
    except FileNotFoundError:
        return
    
    paraphrasing_df = paraphrasing_df[paraphrasing_df['total_sentences_in_chain'] > 0].copy()
    if paraphrasing_df.empty:
        print("  - No valid data with non-empty CoTs found. Skipping analysis.")
        return

    paraphrasing_df['percent_paraphrased'] = (paraphrasing_df['num_sentences_paraphrased'] / paraphrasing_df['total_sentences_in_chain']) * 100

    print("Generating main aggregated plot...")
    plot_single_graph(paraphrasing_df, baseline_df, no_reasoning_df, 'aggregated', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = paraphrasing_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, f'{total_steps}_sentences', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data (<=10 chains).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Paraphrasing plots for LALM results.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--restricted', action='store_true')
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--cross-dataset-plot', action='store_true')
    # Opt-In flags
    parser.add_argument('--show-accuracy-curve', action='store_true')
    parser.add_argument('--show-consistency-curve', action='store_true')
    parser.add_argument('--show-baseline-benchmark', action='store_true')
    parser.add_argument('--show-nr-benchmark', action='store_true')
    args = parser.parse_args()
    
    show_flags = { "show_accuracy": args.show_accuracy_curve, "show_consistency": args.show_consistency_curve, "show_baseline": args.show_baseline_benchmark, "show_nr": args.show_nr_benchmark }
    cross_dataset_show_flags = { "show_accuracy": args.show_accuracy_curve, "show_consistency": args.show_consistency_curve }

    if args.cross_dataset_plot:
        if args.dataset != 'all':
            parser.error("--cross-dataset-plot can only be used with --dataset all")
        if args.grouped:
            parser.error("--cross-dataset-plot cannot be used with --grouped")
        create_cross_dataset_analysis(args.model, args.results_dir, args.plots_dir, args.restricted, args.save_pdf, cross_dataset_show_flags)
    elif args.dataset == 'all':
        try:
            baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            if args.restricted:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
            else:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if not f.endswith('-restricted.jsonl')])))
            
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at {baseline_dir}.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)