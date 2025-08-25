# analysis/compare_transcribed_audio_flipped_partial_filler_text.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from .utils import load_results

def calculate_accuracy(df: pd.DataFrame, pred_col: str, correct_col: str) -> float:
    """
    Calculates accuracy only on the subset of trials where the model provided a valid answer (A, B, C, etc.).
    """
    valid_answers_df = df.dropna(subset=[pred_col])
    valid_answers_df = valid_answers_df[valid_answers_df[pred_col] != "REFUSAL"]
    if valid_answers_df.empty: return 0.0
    return (valid_answers_df[pred_col] == valid_answers_df[correct_col]).mean() * 100

def plot_comparative_graph(df_default: pd.DataFrame, df_transcribed: pd.DataFrame, baseline_default_df: pd.DataFrame, baseline_transcribed_df: pd.DataFrame, no_reasoning_default_df: pd.DataFrame, no_reasoning_transcribed_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
    """
    Generates a single plot comparing the 'flipped_partial_filler_text' results for both conditions.
    """
    # --- Data Integrity: Inner Merge ---
    merge_keys = ['id', 'chain_id', 'percent_replaced']
    combined_df = pd.merge(df_default, df_transcribed, on=merge_keys, suffixes=('_default', '_transcribed'))
    
    if combined_df.empty:
        print(f"  - Skipping plot for '{plot_group_name}' as no common chains were found between conditions.")
        return

    num_chains = len(combined_df[['id', 'chain_id']].drop_duplicates())

    # --- Benchmark Calculation (Condition-Aware and Null-Aware) ---
    relevant_question_ids = combined_df[['id']].drop_duplicates()
    
    rel_bl_def = pd.merge(baseline_default_df, relevant_question_ids, on='id')
    rel_bl_trn = pd.merge(baseline_transcribed_df, relevant_question_ids, on='id')
    rel_nr_def = pd.merge(no_reasoning_default_df, relevant_question_ids, on='id')
    rel_nr_trn = pd.merge(no_reasoning_transcribed_df, relevant_question_ids, on='id')

    bench_bl_def_acc = calculate_accuracy(rel_bl_def, 'predicted_choice', 'correct_choice')
    bench_bl_trn_acc = calculate_accuracy(rel_bl_trn, 'predicted_choice', 'correct_choice')
    bench_nr_def_acc = calculate_accuracy(rel_nr_def, 'predicted_choice', 'correct_choice')
    bench_nr_trn_acc = calculate_accuracy(rel_nr_trn, 'predicted_choice', 'correct_choice')

    # --- Curve Generation with Conditional Binning ---
    if plot_group_name == 'aggregated':
        combined_df['percent_binned'] = (combined_df['percent_replaced'] / 5).round() * 5
        grouping_col = 'percent_binned'
    else:
        grouping_col = 'percent_replaced'

    # --- Accuracy Curves (Nulls Excluded) ---
    acc_default_curve = combined_df.groupby(grouping_col).apply(lambda g: calculate_accuracy(g, 'predicted_choice_default', 'correct_choice_default'), include_groups=False)
    acc_transcribed_curve = combined_df.groupby(grouping_col).apply(lambda g: calculate_accuracy(g, 'predicted_choice_transcribed', 'correct_choice_transcribed'), include_groups=False)

    # --- Consistency Curves (Nulls Included) ---
    con_default_curve = combined_df.groupby(grouping_col)['is_consistent_with_baseline_default'].mean() * 100
    con_transcribed_curve = combined_df.groupby(grouping_col)['is_consistent_with_baseline_transcribed'].mean() * 100

    # The 0% point for the curve is derived directly from the baseline data.
    acc_default_curve[0] = bench_bl_def_acc
    acc_transcribed_curve[0] = bench_bl_trn_acc
    con_default_curve[0] = 100.0
    con_transcribed_curve[0] = 100.0 # Consistency is 100% at 0% replacement by definition for both
    acc_default_curve.sort_index(inplace=True); acc_transcribed_curve.sort_index(inplace=True)
    con_default_curve.sort_index(inplace=True); con_transcribed_curve.sort_index(inplace=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    ax.plot(acc_default_curve.index, acc_default_curve.values, marker='^', linestyle='-', label='Accuracy (Original Audio)')
    ax.plot(con_default_curve.index, con_default_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency (Original Audio)')
    ax.plot(acc_transcribed_curve.index, acc_transcribed_curve.values, marker='^', linestyle='--', color='dodgerblue', label='Accuracy (Transcribed Audio)')
    ax.plot(con_transcribed_curve.index, con_transcribed_curve.values, marker='o', linestyle='--', color='sienna', label='Consistency (Transcribed Audio)')

    ax.axhline(y=bench_nr_def_acc, color='red', linestyle=':', label=f'No-Reasoning Acc (Original) ({bench_nr_def_acc:.2f}%)')
    ax.axhline(y=bench_nr_trn_acc, color='salmon', linestyle=':', label=f'No-Reasoning Acc (Transcribed) ({bench_nr_trn_acc:.2f}%)')
    ax.axhline(y=bench_bl_def_acc, color='green', linestyle='--', label=f'Baseline Acc (Original) ({bench_bl_def_acc:.2f}%)')
    ax.axhline(y=bench_bl_trn_acc, color='lime', linestyle='--', label=f'Baseline Acc (Transcribed) ({bench_bl_trn_acc:.2f}%)')

    # --- Formatting and Saving ---
    base_title = f'Partial Filler (from End) Comparison: Original vs. Transcribed Audio ({dataset_name.upper()})'
    if plot_group_name == 'aggregated':
        subtitle = f'(Aggregated Across {num_chains} Common Chains)'
    else:
        subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Common Chains)'
    ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
    ax.set_xlabel('% of Final Reasoning Replaced by Filler', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105); ax.legend(title='Condition & Metric', loc='best'); fig.tight_layout()

    output_plot_dir = os.path.join(plots_dir, 'comparative_transcribed_audio', 'flipped_partial_filler_text', dataset_name, 'aggregated' if plot_group_name == 'aggregated' else 'grouped')
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"compare_flipped_partial_filler_{dataset_name}_{plot_group_name}.png")
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"  - Plot saved successfully to: {plot_path}")


def create_comparative_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool):
    """ Main function to orchestrate the comparative analysis. """
    print(f"\n--- Generating Comparative Flipped Partial Filler Analysis for: {dataset_name.upper()} ---")
    
    try:
        # Load all 8 required data files.
        default_df = load_results(results_dir, 'flipped_partial_filler_text', dataset_name, 'default')
        transcribed_df = load_results(results_dir, 'flipped_partial_filler_text', dataset_name, 'transcribed_audio')
        baseline_default_df = load_results(results_dir, 'baseline', dataset_name, 'default')
        baseline_transcribed_df = load_results(results_dir, 'baseline', dataset_name, 'transcribed_audio')
        no_reasoning_default_df = load_results(results_dir, 'no_reasoning', dataset_name, 'default')
        no_reasoning_transcribed_df = load_results(results_dir, 'no_reasoning', dataset_name, 'transcribed_audio')
        early_default_df = load_results(results_dir, 'early_answering', dataset_name, 'default')
        early_transcribed_df = load_results(results_dir, 'early_answering', dataset_name, 'transcribed_audio')
    except FileNotFoundError:
        return

    # --- Data Preparation ---
    # Add consistency info by merging with baseline predictions for each condition.
    bl_preds_def = baseline_default_df[['id', 'chain_id', 'predicted_choice']].rename(columns={'predicted_choice': 'baseline_predicted_choice'})
    bl_preds_trn = baseline_transcribed_df[['id', 'chain_id', 'predicted_choice']].rename(columns={'predicted_choice': 'baseline_predicted_choice'})
    default_df = pd.merge(default_df, bl_preds_def, on=['id', 'chain_id'], how='inner')
    transcribed_df = pd.merge(transcribed_df, bl_preds_trn, on=['id', 'chain_id'], how='inner')
    default_df['is_consistent_with_baseline'] = (default_df['predicted_choice'] == default_df['baseline_predicted_choice'])
    transcribed_df['is_consistent_with_baseline'] = (transcribed_df['predicted_choice'] == transcribed_df['baseline_predicted_choice'])

    # Add sentence counts for grouping, merging each condition's data with its corresponding early_answering results.
    sent_counts_def = early_default_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
    sent_counts_trn = early_transcribed_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
    default_df = pd.merge(default_df, sent_counts_def, on=['id', 'chain_id'], how='inner')
    transcribed_df = pd.merge(transcribed_df, sent_counts_trn, on=['id', 'chain_id'], how='inner')

    # Apply the "Meaningful Manipulation" filter to both.
    default_df = default_df[default_df['total_sentences_in_chain'] > 0].copy()
    transcribed_df = transcribed_df[transcribed_df['total_sentences_in_chain'] > 0].copy()
    
    print("Generating main aggregated plot...")
    plot_comparative_graph(default_df, transcribed_df, baseline_default_df, baseline_transcribed_df, no_reasoning_default_df, no_reasoning_transcribed_df, 'aggregated', dataset_name, plots_dir)

    if generate_grouped:
        print("\nGenerating per-length grouped plots...")
        grouped_default = default_df.groupby('total_sentences_in_chain')
        grouped_transcribed = transcribed_df.groupby('total_sentences_in_chain')
        common_lengths = set(grouped_default.groups.keys()) & set(grouped_transcribed.groups.keys())

        for total_steps in sorted(list(common_lengths)):
            group_df_default = grouped_default.get_group(total_steps)
            group_df_transcribed = grouped_transcribed.get_group(total_steps)
            
            if len(group_df_default[['id', 'chain_id']].drop_duplicates()) > 10 and \
               len(group_df_transcribed[['id', 'chain_id']].drop_duplicates()) > 10:
                plot_comparative_graph(group_df_default, group_df_transcribed, baseline_default_df, baseline_transcribed_df, no_reasoning_default_df, no_reasoning_transcribed_df, f'{total_steps}_sentences', dataset_name, plots_dir)
            else:
                print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparative Flipped Partial Filler plots.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--grouped', action='store_true')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            # This logic now correctly finds common datasets across the two condition directories.
            default_dir = os.path.join(args.results_dir, 'flipped_partial_filler_text')
            transcribed_dir = os.path.join(args.results_dir, 'transcribed_audio_experiments', 'flipped_partial_filler_text')
            
            datasets_default, datasets_transcribed = set(), set()
            if os.path.exists(default_dir):
                datasets_default = set([f.replace('flipped_partial_filler_text_', '').replace('.jsonl', '') for f in os.listdir(default_dir) if f.endswith('.jsonl')])
            if os.path.exists(transcribed_dir):
                datasets_transcribed = set([f.replace('flipped_partial_filler_text_', '').replace('_transcribed_audio.jsonl', '') for f in os.listdir(transcribed_dir) if f.endswith('_transcribed_audio.jsonl')])
            
            common_datasets = sorted(list(datasets_default & datasets_transcribed))
            print(f"Found common datasets for comparison: {common_datasets}")
            
            for dataset in common_datasets:
                create_comparative_analysis(dataset, args.results_dir, args.plots_dir, args.grouped)
        except FileNotFoundError:
            print(f"Could not find one of the required 'flipped_partial_filler_text' results directories.")
    else:
        create_comparative_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped)