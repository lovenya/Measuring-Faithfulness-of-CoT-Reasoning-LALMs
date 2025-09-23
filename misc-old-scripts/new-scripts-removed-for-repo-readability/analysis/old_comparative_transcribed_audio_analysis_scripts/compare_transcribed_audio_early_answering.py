# analysis/compare_transcribed_audio_early_answering.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from .utils import load_results

def calculate_accuracy(df: pd.DataFrame, pred_col: str, correct_col: str) -> float:
    """
    Calculates accuracy only on the subset of trials where the model provided a valid answer (A, B, C, etc.).
    This is our definitive, methodologically sound way of measuring accuracy.
    """
    # First, create a clean subset of the data by filtering out any rows where the model
    # refused to answer or gave a null prediction.
    valid_answers_df = df.dropna(subset=[pred_col])
    valid_answers_df = valid_answers_df[valid_answers_df[pred_col] != "REFUSAL"]
    
    if valid_answers_df.empty:
        return 0.0

    # On this clean subset, calculate the percentage of correct answers.
    return (valid_answers_df[pred_col] == valid_answers_df[correct_col]).mean() * 100

def plot_comparative_graph(df_default: pd.DataFrame, df_transcribed: pd.DataFrame, baseline_default_df: pd.DataFrame, baseline_transcribed_df: pd.DataFrame, no_reasoning_default_df: pd.DataFrame, no_reasoning_transcribed_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Generates and saves a single plot comparing the 'default' and 'transcribed_audio' conditions.
    """
    # --- Data Integrity: Inner Merge ---
    # To ensure a fair, "apples-to-apples" comparison, we only analyze the chains
    # that successfully completed in BOTH experimental conditions. This is a critical step for scientific validity.
    merge_keys = ['id', 'chain_id', 'num_sentences_provided']
    combined_df = pd.merge(df_default, df_transcribed, on=merge_keys, suffixes=('_default', '_transcribed'))
    
    if combined_df.empty:
        print(f"  - Skipping plot for '{plot_group_name}' as no common chains were found between conditions.")
        return

    # The number of chains is now calculated from this clean, merged dataset.
    num_chains = len(combined_df[['id', 'chain_id']].drop_duplicates())

    # --- Benchmark Calculation (Condition-Aware and Null-Aware) ---
    # We calculate benchmarks only on the subset of questions present in our merged data.
    relevant_question_ids = combined_df[['id']].drop_duplicates()
    
    rel_bl_def = pd.merge(baseline_default_df, relevant_question_ids, on='id')
    rel_bl_trn = pd.merge(baseline_transcribed_df, relevant_question_ids, on='id')
    rel_nr_def = pd.merge(no_reasoning_default_df, relevant_question_ids, on='id')
    rel_nr_trn = pd.merge(no_reasoning_transcribed_df, relevant_question_ids, on='id')

    # We use our robust, null-excluding function to calculate all benchmark accuracies.
    bench_bl_def_acc = calculate_accuracy(rel_bl_def, 'predicted_choice', 'correct_choice')
    bench_bl_trn_acc = calculate_accuracy(rel_bl_trn, 'predicted_choice', 'correct_choice')
    bench_nr_def_acc = calculate_accuracy(rel_nr_def, 'predicted_choice', 'correct_choice')
    bench_nr_trn_acc = calculate_accuracy(rel_nr_trn, 'predicted_choice', 'correct_choice')

    # --- Curve Generation with Conditional Binning ---
    if plot_group_name == 'aggregated':
        combined_df['percent_binned'] = (combined_df['percent_reasoning_provided_default'] / 5).round() * 5
        grouping_col = 'percent_binned'
    else:
        grouping_col = 'percent_reasoning_provided_default'

    # --- Accuracy Curves (Nulls Excluded) ---
    acc_default_curve = combined_df.groupby(grouping_col).apply(lambda g: calculate_accuracy(g, 'predicted_choice_default', 'correct_choice_default'))
    acc_transcribed_curve = combined_df.groupby(grouping_col).apply(lambda g: calculate_accuracy(g, 'predicted_choice_transcribed', 'correct_choice_transcribed'))

    # --- Consistency Curves (Nulls Included) ---
    # For consistency, a REFUSAL or null is a valid state, so we use the full, unfiltered data.
    con_default_curve = combined_df.groupby(grouping_col)['is_consistent_with_baseline_default'].mean() * 100
    con_transcribed_curve = combined_df.groupby(grouping_col)['is_consistent_with_baseline_transcribed'].mean() * 100

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot the 'default' condition curves with SOLID lines.
    ax.plot(acc_default_curve.index, acc_default_curve.values, marker='^', linestyle='-', label='Accuracy (Original Audio)')
    ax.plot(con_default_curve.index, con_default_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency (Original Audio)')

    # Plot the 'transcribed_audio' condition curves with DASHED lines of the same colors.
    ax.plot(acc_transcribed_curve.index, acc_transcribed_curve.values, marker='^', linestyle='--', color='dodgerblue', label='Accuracy (Transcribed Audio)')
    ax.plot(con_transcribed_curve.index, con_transcribed_curve.values, marker='o', linestyle='--', color='sienna', label='Consistency (Transcribed Audio)')

    # Plot the four benchmark lines for a rich comparison.
    ax.axhline(y=bench_nr_def_acc, color='red', linestyle=':', label=f'No-Reasoning Acc (Original) ({bench_nr_def_acc:.2f}%)')
    ax.axhline(y=bench_nr_trn_acc, color='salmon', linestyle=':', label=f'No-Reasoning Acc (Transcribed) ({bench_nr_trn_acc:.2f}%)')
    ax.axhline(y=bench_bl_def_acc, color='green', linestyle='--', label=f'Baseline Acc (Original) ({bench_bl_def_acc:.2f}%)')
    ax.axhline(y=bench_bl_trn_acc, color='lime', linestyle='--', label=f'Baseline Acc (Transcribed) ({bench_bl_trn_acc:.2f}%)')

    # --- Formatting and Saving ---
    base_title = f'Early Answering Comparison: Original vs. Transcribed Audio ({dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Common Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Common Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
        
    ax.set_xlabel('% of Reasoning Chain Provided', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Condition & Metric', loc='best'); fig.tight_layout()

    # Save the plot to our standard comparative directory structure.
    if plot_group_name == 'aggregated':
        output_plot_dir = os.path.join(plots_dir, 'comparative_transcribed_audio', 'early_answering', dataset_name, 'aggregated')
    else:
        output_plot_dir = os.path.join(plots_dir, 'comparative_transcribed_audio', 'early_answering', dataset_name, 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"compare_early_answering_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


def create_comparative_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool):
    """ Main function to orchestrate the comparative analysis for the Early Answering experiment. """
    print(f"\n--- Generating Comparative Early Answering Analysis for: {dataset_name.upper()} ---")
    
    try:
        # Load data for BOTH conditions, plus all necessary benchmarks.
        early_default_df = load_results(results_dir, 'early_answering', dataset_name, 'default')
        early_transcribed_df = load_results(results_dir, 'early_answering', dataset_name, 'transcribed_audio')
        baseline_default_df = load_results(results_dir, 'baseline', dataset_name, 'default')
        baseline_transcribed_df = load_results(results_dir, 'baseline', dataset_name, 'transcribed_audio')
        no_reasoning_default_df = load_results(results_dir, 'no_reasoning', dataset_name, 'default')
        no_reasoning_transcribed_df = load_results(results_dir, 'no_reasoning', dataset_name, 'transcribed_audio')
    except FileNotFoundError:
        return

    # --- Data Preparation for Both DataFrames ---
    # Apply the "Meaningful Manipulation" filter: CoTs must have at least one sentence.
    early_default_df = early_default_df[early_default_df['total_sentences_in_chain'] > 0].copy()
    early_transcribed_df = early_transcribed_df[early_transcribed_df['total_sentences_in_chain'] > 0].copy()
    
    # Calculate the primary x-axis variable for both.
    early_default_df['percent_reasoning_provided'] = (early_default_df['num_sentences_provided'] / early_default_df['total_sentences_in_chain']) * 100
    early_transcribed_df['percent_reasoning_provided'] = (early_transcribed_df['num_sentences_provided'] / early_transcribed_df['total_sentences_in_chain']) * 100
    
    # Generate the main aggregated plot.
    print("Generating main aggregated plot...")
    plot_comparative_graph(early_default_df, early_transcribed_df, baseline_default_df, baseline_transcribed_df, no_reasoning_default_df, no_reasoning_transcribed_df, 'aggregated', dataset_name, plots_dir)

    # Optionally, generate the more detailed per-length plots.
    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_default = early_default_df.groupby('total_sentences_in_chain')
        grouped_transcribed = early_transcribed_df.groupby('total_sentences_in_chain')
        
        common_lengths = set(grouped_default.groups.keys()) & set(grouped_transcribed.groups.keys())

        for total_steps in sorted(list(common_lengths)):
            group_df_default = grouped_default.get_group(total_steps)
            group_df_transcribed = grouped_transcribed.get_group(total_steps)
            
            if len(group_df_default[['id', 'chain_id']].drop_duplicates()) > 10 and \
               len(group_df_transcribed[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_comparative_graph(group_df_default, group_df_transcribed, baseline_default_df, baseline_transcribed_df, no_reasoning_default_df, no_reasoning_transcribed_df, f'{total_steps}_sentences', dataset_name, plots_dir)
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
       
        # This block scans the two separate directories to find the
        # result files for each condition, then finds the intersection.
        try:
            # Define the two distinct directories to scan.
            default_dir = os.path.join(args.results_dir, 'early_answering')
            transcribed_dir = os.path.join(args.results_dir, 'transcribed_audio_experiments', 'early_answering')

            # Scan the default directory for its completed datasets.
            datasets_default = set()
            if os.path.exists(default_dir):
                datasets_default = set([
                    f.replace('early_answering_', '').replace('.jsonl', '') 
                    for f in os.listdir(default_dir) if f.endswith('.jsonl')
                ])

            # Scan the transcribed directory for its completed datasets.
            datasets_transcribed = set()
            if os.path.exists(transcribed_dir):
                datasets_transcribed = set([
                    f.replace('early_answering_', '').replace('_transcribed_audio.jsonl', '') 
                    for f in os.listdir(transcribed_dir) if f.endswith('_transcribed_audio.jsonl')
                ])
            
            # The common datasets are the intersection of the two sets.
            common_datasets = sorted(list(datasets_default & datasets_transcribed))
            
            if not common_datasets:
                print("No common datasets found with completed runs for both 'default' and 'transcribed_audio' conditions.")
            else:
                print(f"Found common datasets for comparison: {common_datasets}")
                for dataset in common_datasets:
                    create_comparative_analysis(dataset, args.results_dir, args.plots_dir, args.grouped)
        
        except FileNotFoundError:
            print(f"Could not find one of the required results directories. Please check paths.")
       
    else:
        create_comparative_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped)