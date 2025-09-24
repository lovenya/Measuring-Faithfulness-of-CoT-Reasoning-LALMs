# analysis/plot_early_answering.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, plot_group_name: str, model_name: str, dataset_name: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_accuracy: bool, show_consistency: bool, show_baseline: bool, show_nr: bool):
    """
    Generates a single plot for 'Early Answering' data, with full feature control.
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
        df['percent_binned'] = (df['percent_reasoning_provided'] / 5).round() * 5
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        accuracy_curve = df.groupby('percent_reasoning_provided')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_reasoning_provided')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    if show_accuracy:
        ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy at Step')
    if show_consistency:
        ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Final Answer')
    if show_nr:
        ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if show_baseline:
        ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Final CoT Accuracy ({baseline_accuracy:.2f}%)')

    restriction_str = " (Restricted)" if is_restricted else " (Full Dataset)"
    base_title = f'Accuracy & Consistency vs. Reasoning Progression ({model_name.upper()} on {dataset_name.upper()}){restriction_str}'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Provided', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Output Path ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, model_name, 'early_answering', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, model_name, 'early_answering', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"early_answering_{model_name}_{dataset_name}_{plot_group_name}{suffix}"
    
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
    df['percent_binned'] = (df['percent_reasoning_provided'] / 5).round() * 5

    if show_accuracy:
        accuracy_curves = df.groupby(['dataset', 'percent_binned'])['is_correct'].mean().reset_index()
        accuracy_curves['is_correct'] *= 100
        sns.lineplot(data=accuracy_curves, x='percent_binned', y='is_correct', hue='dataset', ax=ax, marker='^', linestyle='--', palette=palette, legend=False)

    if show_consistency:
        consistency_curves = df.groupby(['dataset', 'percent_binned'])['is_consistent_with_baseline'].mean().reset_index()
        consistency_curves['is_consistent_with_baseline'] *= 100
        sns.lineplot(data=consistency_curves, x='percent_binned', y='is_consistent_with_baseline', hue='dataset', ax=ax, marker='o', linestyle='-', palette=palette)

    plotted_metrics = []
    if show_accuracy: plotted_metrics.append("Accuracy")
    if show_consistency: plotted_metrics.append("Consistency")
    
    restriction_str = " (Restricted)" if is_restricted else " (Full Dataset)"
    ax.set_title(f'Cross-Dataset Comparison of { " & ".join(plotted_metrics) }: Early Answering ({model_name.upper()}){restriction_str}', fontsize=16, pad=20)
    
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = []
    if show_accuracy:
        legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', marker='^', label='Accuracy'))
    if show_consistency:
        legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', marker='o', label='Consistency'))
    
    dataset_handles = [h for h in handles if isinstance(h, Line2D)]
    dataset_labels = labels[:df['dataset'].nunique()]
    
    l1 = ax.legend(handles=dataset_handles, labels=dataset_labels, title="Dataset", loc="upper right")
    ax.add_artist(l1)
    if legend_elements:
        ax.legend(handles=legend_elements, title="Metric", loc="upper left")

    ax.set_xlabel('% of Reasoning Chain Provided', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); fig.tight_layout()

    output_plot_dir = os.path.join(plots_dir, model_name, 'early_answering', 'cross_dataset')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"cross_dataset_early_answering_{model_name}{suffix}"
    
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
    print(f"\n--- Generating CROSS-DATASET Early Answering Analysis for: {model_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
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
                df = load_results(model_name, results_dir, 'early_answering', dataset, is_restricted)
                df = df[df['total_sentences_in_chain'] > 0].copy()
                if not df.empty:
                    df['percent_reasoning_provided'] = (df['num_sentences_provided'] / df['total_sentences_in_chain']) * 100
                    df['dataset'] = dataset
                    all_dfs.append(df)
                else:
                    print(f"  - WARNING: No valid data for '{dataset}' in early_answering results. Skipping.")
            except FileNotFoundError:
                print(f"  - WARNING: 'early_answering' results for dataset '{dataset}' not found. Skipping.")
                continue
        
        if not all_dfs:
            print("No data found for any dataset. Halting analysis.")
            return
            
        super_df = pd.concat(all_dfs, ignore_index=True)
        plot_cross_dataset_graph(super_df, model_name, plots_dir, is_restricted, save_as_pdf, **show_flags)

    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}' at {baseline_dir}. Cannot run for 'all' datasets.")
        return

def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, is_restricted: bool, generate_grouped: bool, save_as_pdf: bool, show_flags: dict):
    """ Main function to orchestrate the early answering analysis. """
    print(f"\n--- Generating Early Answering Analysis for: {model_name.upper()} on {dataset_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name, is_restricted)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name, is_restricted)
        early_df = load_results(model_name, results_dir, 'early_answering', dataset_name, is_restricted)
    except FileNotFoundError:
        return
    
    early_df = early_df[early_df['total_sentences_in_chain'] > 0].copy()
    if early_df.empty:
        print("  - No valid data with non-empty CoTs found. Skipping analysis.")
        return

    early_df['percent_reasoning_provided'] = (early_df['num_sentences_provided'] / early_df['total_sentences_in_chain']) * 100

    print("Generating main aggregated plot...")
    plot_single_graph(early_df, baseline_df, no_reasoning_df, 'aggregated', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = early_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, f'{total_steps}_sentences', model_name, dataset_name, plots_dir, is_restricted, save_as_pdf, **show_flags)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data (<=10 chains).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or extract data from Early Answering results.")
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
    parser.add_argument('--print-line-data', action='store_true', help="Calculate and print the AGGREGATED line plot data to the console and exit.")
    
    args = parser.parse_args()
    
    # --- NEW, CORRECTED LOGIC BLOCK FOR DATA EXTRACTION ---
    if args.print_line_data:
        if not args.cross_dataset_plot:
            parser.error("--print-line-data is currently only supported for --cross-dataset-plot mode.")

        print(f"--- Extracting AGGREGATED Line Plot Data for: {args.model.upper()} on ALL datasets {' (Restricted)' if args.restricted else ''} ---")
        
        all_dfs = []
        try:
            baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            if args.restricted:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('-restricted.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('-restricted.jsonl')])))
            else:
                dataset_names = sorted(list(set([f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if not f.endswith('-restricted.jsonl')])))
            
            for dataset in dataset_names:
                try:
                    df = load_results(args.model, args.results_dir, 'early_answering', dataset, args.restricted)
                    df = df[df['total_sentences_in_chain'] > 0].copy()
                    if not df.empty:
                        df['percent_reasoning_provided'] = (df['num_sentences_provided'] / df['total_sentences_in_chain']) * 100
                        df['dataset'] = dataset
                        all_dfs.append(df)
                except FileNotFoundError:
                    continue
            
            if not all_dfs:
                print("No data found for any dataset. Halting.")
                exit()
                
            super_df = pd.concat(all_dfs, ignore_index=True)
            super_df['percent_binned'] = (super_df['percent_reasoning_provided'] / 5).round() * 5

            print("\n") # Spacer for readability

            # Loop through each dataset and print its aggregated line data
            for dataset_name, group_df in super_df.groupby('dataset'):
                print("="*60)
                print(f"Dataset: {dataset_name}")
                print("="*60)
                
                if args.show_accuracy_curve:
                    accuracy_curve = group_df.groupby('percent_binned')['is_correct'].mean() * 100
                    print("\nAccuracy Curve Coordinates:")
                    print(f"  X Coords: {accuracy_curve.index.tolist()}")
                    print(f"  Y Coords: {accuracy_curve.values.tolist()}")

                if args.show_consistency_curve:
                    consistency_curve = group_df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
                    print("\nConsistency Curve Coordinates:")
                    print(f"  X Coords: {consistency_curve.index.tolist()}")
                    print(f"  Y Coords: {consistency_curve.values.tolist()}")
                
                print("\n")

            exit()

        except Exception as e:
            print(f"An error occurred during data extraction: {e}")
            exit(1)

    # If --print-scatter-data is not used, the script proceeds with the normal plotting logic.
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