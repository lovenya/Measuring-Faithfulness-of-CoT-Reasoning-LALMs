# analysis/comparative_transcribed_audio_analysis/compare_early_answering.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from analysis.utils import load_results

def plot_comparative_graph(df_default: pd.DataFrame, df_transcribed: pd.DataFrame, baseline_df: pd.DataFrame, no_reasoning_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Generates a single plot comparing the 'default' and 'transcribed_audio' conditions.
    """
    # --- Data Integrity: Inner Merge ---
    merge_keys = ['id', 'chain_id', 'num_sentences_provided']
    # We now merge on the raw dataframes and calculate percentages *after* the merge.
    combined_df = pd.merge(df_default, df_transcribed, on=merge_keys, suffixes=('_default', '_transcribed'))
    
    if combined_df.empty:
        print(f"  - Skipping plot for '{plot_group_name}' as no common chains were found between conditions.")
        return

    num_chains = len(combined_df[['id', 'chain_id']].drop_duplicates())

    # --- Benchmark Calculation ---
    relevant_question_ids = combined_df[['id']].drop_duplicates()
    relevant_baseline_df = pd.merge(baseline_df, relevant_question_ids, on='id')
    relevant_no_reasoning_df = pd.merge(no_reasoning_df, relevant_question_ids, on='id')
    baseline_accuracy = relevant_baseline_df.groupby('id')['is_correct'].mean().mean() * 100
    no_reasoning_accuracy = relevant_no_reasoning_df.groupby('id')['is_correct'].mean().mean() * 100

    # --- THE CRITICAL FIX (Part 1): Calculate percentages on the merged dataframe ---
    combined_df['percent_reasoning_provided_default'] = (combined_df['num_sentences_provided'] / combined_df['total_sentences_in_chain_default']) * 100

    # --- Curve Generation with Conditional Binning ---
    if plot_group_name == 'aggregated':
        combined_df['percent_binned'] = (combined_df['percent_reasoning_provided_default'] / 5).round() * 5
        grouping_col = 'percent_binned'
    else:
        grouping_col = 'percent_reasoning_provided_default'

    acc_default = combined_df.groupby(grouping_col)['is_correct_default'].mean() * 100
    con_default = combined_df.groupby(grouping_col)['is_consistent_with_baseline_default'].mean() * 100
    acc_transcribed = combined_df.groupby(grouping_col)['is_correct_transcribed'].mean() * 100
    con_transcribed = combined_df.groupby(grouping_col)['is_consistent_with_baseline_transcribed'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    ax.plot(acc_default.index, acc_default.values, marker='^', linestyle='-', label='Accuracy (Original Audio)')
    ax.plot(con_default.index, con_default.values, marker='o', linestyle='-', color='#8c564b', label='Consistency (Original Audio)')
    ax.plot(acc_transcribed.index, acc_transcribed.values, marker='^', linestyle='--', color='dodgerblue', label='Accuracy (Transcribed Audio)')
    ax.plot(con_transcribed.index, con_transcribed.values, marker='o', linestyle='--', color='sienna', label='Consistency (Transcribed Audio)')
    ax.axhline(y=no_reasoning_accuracy, color='red', linestyle=':', label=f'No-Reasoning Accuracy ({no_reasoning_accuracy:.2f}%)')
    ax.axhline(y=baseline_accuracy, color='green', linestyle=':', label=f'Final CoT Accuracy ({baseline_accuracy:.2f}%)')

    base_title = f'Early Answering Comparison: Original vs. Transcribed Audio ({dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Common Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Common Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Provided', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Condition & Metric', loc='best'); fig.tight_layout()

    # --- Save Figure ---
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, 'comparative_transcribed_audio', 'early_answering', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, 'comparative_transcribed_audio', 'early_answering', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"compare_early_answering_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


def create_comparative_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool):
    """ Main function to orchestrate the comparative analysis. """
    print(f"\n--- Generating Comparative Early Answering Analysis for: {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(results_dir, 'baseline', dataset_name, 'default')
        no_reasoning_df = load_results(results_dir, 'no_reasoning', dataset_name, 'default')
        early_default_df = load_results(results_dir, 'early_answering', dataset_name, 'default')
        early_transcribed_df = load_results(results_dir, 'early_answering', dataset_name, 'transcribed_audio')
    except FileNotFoundError:
        return

    # --- THE CRITICAL FIX (Part 2): Correctly prepare the dataframes BEFORE passing them ---
    # Filter out zero-sentence CoTs first, as they are invalid for this analysis.
    early_default_df = early_default_df[early_default_df['total_sentences_in_chain'] > 0].copy()
    early_transcribed_df = early_transcribed_df[early_transcribed_df['total_sentences_in_chain'] > 0].copy()

    # Generate the main aggregated plot.
    print("Generating main aggregated plot...")
    plot_comparative_graph(early_default_df, early_transcribed_df, baseline_df, no_reasoning_df, 'aggregated', dataset_name, plots_dir)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_default = early_default_df.groupby('total_sentences_in_chain')
        grouped_transcribed = early_transcribed_df.groupby('total_sentences_in_chain')
        all_lengths = set(grouped_default.groups.keys()) & set(grouped_transcribed.groups.keys())

        for total_steps in sorted(list(all_lengths)):
            group_df_default = grouped_default.get_group(total_steps)
            group_df_transcribed = grouped_transcribed.get_group(total_steps)
            
            if len(group_df_default[['id', 'chain_id']].drop_duplicates()) > 10 and \
               len(group_df_transcribed[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_comparative_graph(group_df_default, group_df_transcribed, baseline_df, no_reasoning_df, f'{total_steps}_sentences', dataset_name, plots_dir)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data in one or both conditions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparative Early Answering plots.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--grouped', action='store_true')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            default_dir = os.path.join(args.results_dir, 'early_answering')
            datasets_default = set([f.replace('early_answering_', '').replace('_default.jsonl', '') for f in os.listdir(default_dir) if f.endswith('_default.jsonl')])
            datasets_transcribed = set([f.replace('early_answering_', '').replace('_transcribed_audio.jsonl', '') for f in os.listdir(default_dir) if f.endswith('_transcribed_audio.jsonl')])
            common_datasets = sorted(list(datasets_default & datasets_transcribed))
            print(f"Found common datasets for comparison: {common_datasets}")
            for dataset in common_datasets:
                create_comparative_analysis(dataset, args.results_dir, args.plots_dir, args.grouped)
        except FileNotFoundError:
            print(f"Could not find 'early_answering' results directory at {default_dir}.")
    else:
        create_comparative_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped)