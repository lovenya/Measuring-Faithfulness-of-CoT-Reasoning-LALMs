# analysis/plot_adding_mistakes.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_results

def plot_single_graph(df: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, plot_group_name: str, model_name: str, dataset_name: str, plots_dir: str, save_as_pdf: bool, show_accuracy: bool, show_consistency: bool, show_baseline: bool, show_nr: bool):
    """
    Generates a single plot for 'Adding Mistakes' data, with full feature control.
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
        df['percent_binned'] = (df['percent_before_mistake'] / 10).round() * 10
        accuracy_curve = df.groupby('percent_binned')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_binned')['is_consistent_with_baseline'].mean() * 100
    else:
        accuracy_curve = df.groupby('percent_before_mistake')['is_correct'].mean() * 100
        consistency_curve = df.groupby('percent_before_mistake')['is_consistent_with_baseline'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))

    if show_accuracy:
        ax.plot(accuracy_curve.index, accuracy_curve.values, marker='^', linestyle='--', label='Accuracy After Mistake')
    if show_consistency:
        ax.plot(consistency_curve.index, consistency_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency with Original Answer')
    if show_nr:
        ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    if show_baseline:
        ax.axhline(y=baseline_accuracy, color='green', linestyle='--', label=f'Original CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Accuracy & Consistency vs. Position of Introduced Mistake ({model_name.upper()} on {dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Before Mistake', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Metrics', loc='best'); fig.tight_layout()

    # --- Model-Agnostic & Restriction-Aware Output Path ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, model_name, 'adding_mistakes', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, model_name, 'adding_mistakes', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    base_filename = f"adding_mistakes_{model_name}_{dataset_name}_{plot_group_name}"
    
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
    
    # Use a seaborn color palette for distinct colors for each dataset line
    palette = sns.color_palette("viridis", n_colors=df['dataset'].nunique())

    if show_accuracy:
        # Group by dataset, then calculate the binned accuracy curve for each
        accuracy_curves = df.groupby('dataset').apply(
            lambda x: x.groupby((x['percent_before_mistake'] / 10).round() * 10)['is_correct'].mean() * 100
        ).reset_index()
        sns.lineplot(data=accuracy_curves, x='level_1', y='is_correct', hue='dataset', ax=ax, marker='^', linestyle='--', palette=palette)

    if show_consistency:
        # Group by dataset, then calculate the binned consistency curve for each
        consistency_curves = df.groupby('dataset').apply(
            lambda x: x.groupby((x['percent_before_mistake'] / 10).round() * 10)['is_consistent_with_baseline'].mean() * 100
        ).reset_index()
        sns.lineplot(data=consistency_curves, x='level_1', y='is_consistent_with_baseline', hue='dataset', ax=ax, marker='o', linestyle='-', palette=palette)

    restriction_str = " (Restricted 1-6 Sentences)" if is_restricted else " (Full Dataset)"
    ax.set_title(f'Cross-Dataset Comparison: Adding Mistakes ({model_name.upper()}){restriction_str}', fontsize=16, pad=20)
    ax.set_xlabel('% of Reasoning Chain Before Mistake', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105)
    
    # Improve legend handling for potentially many datasets
    if show_accuracy and show_consistency:
        # Manually create a clear legend if both are shown
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        # This is complex, so for now we simplify
        ax.legend(title='Dataset')
    else:
         ax.legend(title='Dataset')

    fig.tight_layout()

    # --- Output Path ---
    output_plot_dir = os.path.join(plots_dir, model_name, 'adding_mistakes', 'cross_dataset')
    os.makedirs(output_plot_dir, exist_ok=True)
    
    suffix = "-restricted" if is_restricted else ""
    base_filename = f"cross_dataset_adding_mistakes_{model_name}{suffix}"
    
    png_path = os.path.join(output_plot_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Cross-dataset plot saved to: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_plot_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()

# --- NEW MAIN FUNCTION FOR CROSS-DATASET ANALYSIS ---
def create_cross_dataset_analysis(model_name: str, results_dir: str, plots_dir: str, is_restricted: bool, save_as_pdf: bool, show_flags: dict):
    """
    Loads data from ALL datasets and generates a single comparative plot.
    """
    print(f"\n--- Generating CROSS-DATASET Adding Mistakes Analysis for: {model_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    all_dfs = []
    try:
        exp_dir = os.path.join(results_dir, model_name, 'adding_mistakes')
        if is_restricted:
            dataset_names = sorted(list(set([f.replace(f'adding_mistakes_{model_name}_', '').replace('-restricted.jsonl', '') for f in os.listdir(exp_dir) if f.endswith('-restricted.jsonl')])))
        else:
            dataset_names = sorted(list(set([f.replace(f'adding_mistakes_{model_name}_', '').replace('.jsonl', '') for f in os.listdir(exp_dir) if not f.endswith('-restricted.jsonl')])))
        
        print(f"Found datasets: {dataset_names}")

        for dataset in dataset_names:
            df = load_results(model_name, results_dir, 'adding_mistakes', dataset, is_restricted)
            df = df[df['total_sentences_in_chain'] > 0].copy()
            if not df.empty:
                df['percent_before_mistake'] = ((df['mistake_position'] - 1) / df['total_sentences_in_chain']) * 100
                df['dataset'] = dataset # Tag each row with its source dataset
                all_dfs.append(df)
        
        if not all_dfs:
            print("No data found for any dataset. Halting analysis.")
            return
            
        super_df = pd.concat(all_dfs, ignore_index=True)
        plot_cross_dataset_graph(super_df, model_name, plots_dir, is_restricted, save_as_pdf, **show_flags)

    except FileNotFoundError:
        print(f"Could not find directory for model '{model_name}' at {exp_dir}.")
        return

def create_analysis(model_name: str, dataset_name: str, results_dir: str, plots_dir: str, is_restricted: bool, generate_grouped: bool, save_as_pdf: bool, show_flags: dict):
    """ Main function to orchestrate the 'Adding Mistakes' analysis. """
    print(f"\n--- Generating Adding Mistakes Analysis for: {model_name.upper()} on {dataset_name.upper()}{' (Restricted)' if is_restricted else ''} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name, is_restricted)
        no_reasoning_df = load_results(model_name, results_dir, 'no_reasoning', dataset_name, is_restricted)
        mistakes_df = load_results(model_name, results_dir, 'adding_mistakes', dataset_name, is_restricted)
    except FileNotFoundError:
        return

    # --- "Meaningful Manipulation" Filter ---
    mistakes_df = mistakes_df[mistakes_df['total_sentences_in_chain'] > 0].copy()
    if mistakes_df.empty:
        print("  - No valid data with non-empty CoTs found. Skipping analysis.")
        return

    mistakes_df['percent_before_mistake'] = ((mistakes_df['mistake_position'] - 1) / mistakes_df['total_sentences_in_chain']) * 100

    print("Generating main aggregated plot...")
    plot_single_graph(mistakes_df, baseline_df, no_reasoning_df, 'aggregated', model_name, dataset_name, plots_dir, save_as_pdf, **show_flags)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_by_total_steps = mistakes_df.groupby('total_sentences_in_chain')
        for total_steps, group_df in grouped_by_total_steps:
            if len(group_df[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_single_graph(group_df, baseline_df, no_reasoning_df, f'{total_steps}_sentences', model_name, dataset_name, plots_dir, save_as_pdf, **show_flags)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data (<=10 chains).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Adding Mistakes plots for LALM results.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--restricted', action='store_true', help="Analyze the '-restricted.jsonl' files.")
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--save-pdf', action='store_true')
    # Opt-In flags for plot elements
    parser.add_argument('--show-accuracy-curve', action='store_true')
    parser.add_argument('--show-consistency-curve', action='store_true')
    parser.add_argument('--show-baseline-benchmark', action='store_true')
    parser.add_argument('--show-nr-benchmark', action='store_true')
    
    parser.add_argument('--cross-dataset-plot', action='store_true', help="Generate a single plot comparing all datasets.")
    args = parser.parse_args()
    
    show_flags = {
        "show_accuracy": args.show_accuracy_curve,
        "show_consistency": args.show_consistency_curve,
        "show_baseline": args.show_baseline_benchmark,
        "show_nr": args.show_nr_benchmark
    }


    if args.cross_dataset_plot:
        if args.dataset != 'all':
            parser.error("--cross-dataset-plot can only be used with --dataset all")
        if args.grouped:
            parser.error("--cross-dataset-plot cannot be used with --grouped")
        create_cross_dataset_analysis(args.model, args.results_dir, args.plots_dir, args.restricted, args.save_pdf, show_flags)
    
    elif args.dataset == 'all':
        try:
            exp_dir = os.path.join(args.results_dir, args.model, 'adding_mistakes')
            # Logic to discover available datasets, correctly handling the -restricted suffix
            if args.restricted:
                dataset_names = sorted(list(set([f.replace(f'adding_mistakes_{args.model}_', '').replace('-restricted.jsonl', '') for f in os.listdir(exp_dir) if f.endswith('-restricted.jsonl')])))
            else:
                dataset_names = sorted(list(set([f.replace(f'adding_mistakes_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(exp_dir) if not f.endswith('-restricted.jsonl')])))
            
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)
        except FileNotFoundError:
            print(f"Could not find directory for model '{args.model}' at {exp_dir}.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.plots_dir, args.restricted, args.grouped, args.save_pdf, show_flags)