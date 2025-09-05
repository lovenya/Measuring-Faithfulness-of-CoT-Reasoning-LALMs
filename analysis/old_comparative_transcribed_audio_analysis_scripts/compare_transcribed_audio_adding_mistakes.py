# # analysis/compare_transcribed_audio_adding_mistakes.py

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# from .utils import load_results # type: ignore

# def calculate_accuracy(df: pd.DataFrame, pred_col: str, correct_col: str) -> pd.Series:
#     """
#     Calculates accuracy only on the subset of trials where the model provided a valid answer.
#     """
#     # Filter out rows where the model refused to answer or gave a null prediction.
#     valid_answers_df = df.dropna(subset=[pred_col])
#     valid_answers_df = valid_answers_df[valid_answers_df[pred_col] != "REFUSAL"]
    
#     # On this clean subset, calculate correctness.
#     return (valid_answers_df[pred_col] == valid_answers_df[correct_col]).mean() * 100

# def plot_comparative_graph(df_default: pd.DataFrame, df_transcribed: pd.DataFrame, baseline_default_df: pd.DataFrame, baseline_transcribed_df: pd.DataFrame, no_reasoning_default_df: pd.DataFrame, no_reasoning_transcribed_df: pd.DataFrame, plot_group_name: str, dataset_name: str, plots_dir: str):
#     """
#     Generates a single plot comparing the 'adding_mistakes' results for both conditions.
#     """
#     # --- Data Integrity: Inner Merge ---
#     # We only analyze trials that exist in BOTH result sets for a fair comparison.
#     merge_keys = ['id', 'chain_id', 'mistake_position']
#     combined_df = pd.merge(df_default, df_transcribed, on=merge_keys, suffixes=('_default', '_transcribed'))
    
#     if combined_df.empty:
#         print(f"  - Skipping plot for '{plot_group_name}' due to no common trials between conditions.")
#         return

#     num_chains = len(combined_df[['id', 'chain_id']].drop_duplicates())

#     # --- Benchmark Calculation (Condition-Aware and Null-Aware) ---
#     relevant_question_ids = combined_df[['id']].drop_duplicates()
    
#     # Filter each benchmark dataframe to only the relevant questions.
#     rel_bl_def = pd.merge(baseline_default_df, relevant_question_ids, on='id')
#     rel_bl_trn = pd.merge(baseline_transcribed_df, relevant_question_ids, on='id')
#     rel_nr_def = pd.merge(no_reasoning_default_df, relevant_question_ids, on='id')
#     rel_nr_trn = pd.merge(no_reasoning_transcribed_df, relevant_question_ids, on='id')

#     # Calculate accuracy for each benchmark using our robust, null-excluding function.
#     bench_bl_def_acc = calculate_accuracy(rel_bl_def, 'predicted_choice', 'correct_choice')
#     bench_bl_trn_acc = calculate_accuracy(rel_bl_trn, 'predicted_choice', 'correct_choice')
#     bench_nr_def_acc = calculate_accuracy(rel_nr_def, 'predicted_choice', 'correct_choice')
#     bench_nr_trn_acc = calculate_accuracy(rel_nr_trn, 'predicted_choice', 'correct_choice')

#     # --- Data Filtering and Curve Generation ---
#     df_filtered = combined_df[combined_df['percent_before_mistake_default'] <= 90].copy()
#     if df_filtered.empty: return

#     if plot_group_name == 'aggregated':
#         df_filtered['percent_binned'] = (df_filtered['percent_before_mistake_default'] / 10).round() * 10
#         grouping_col = 'percent_binned'
#     else:
#         grouping_col = 'percent_before_mistake_default'

#     # --- Accuracy Curves (Nulls Excluded) ---
#     acc_default_curve = df_filtered.groupby(grouping_col).apply(lambda g: calculate_accuracy(g, 'predicted_choice_default', 'correct_choice_default'))
#     acc_transcribed_curve = df_filtered.groupby(grouping_col).apply(lambda g: calculate_accuracy(g, 'predicted_choice_transcribed', 'correct_choice_transcribed'))

#     # --- Consistency Curves (Nulls Included) ---
#     df_filtered['is_consistent_default'] = (df_filtered['predicted_choice_default'] == df_filtered['corresponding_baseline_predicted_choice_default'])
#     df_filtered['is_consistent_transcribed'] = (df_filtered['predicted_choice_transcribed'] == df_filtered['corresponding_baseline_predicted_choice_transcribed'])
#     con_default_curve = df_filtered.groupby(grouping_col)['is_consistent_default'].mean() * 100
#     con_transcribed_curve = df_filtered.groupby(grouping_col)['is_consistent_transcribed'].mean() * 100

#     # --- Plotting ---
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(14, 9))

#     # Plot the four main curves using our solid-vs-dashed convention.
#     ax.plot(acc_default_curve.index, acc_default_curve.values, marker='^', linestyle='-', label='Accuracy (Original Audio)')
#     ax.plot(con_default_curve.index, con_default_curve.values, marker='o', linestyle='-', color='#8c564b', label='Consistency (Original Audio)')
#     ax.plot(acc_transcribed_curve.index, acc_transcribed_curve.values, marker='^', linestyle='--', color='dodgerblue', label='Accuracy (Transcribed Audio)')
#     ax.plot(con_transcribed_curve.index, con_transcribed_curve.values, marker='o', linestyle='--', color='sienna', label='Consistency (Transcribed Audio)')

#     # Plot the four benchmark lines.
#     ax.axhline(y=bench_nr_def_acc, color='red', linestyle=':', label=f'No-Reasoning Acc (Original) ({bench_nr_def_acc:.2f}%)')
#     ax.axhline(y=bench_nr_trn_acc, color='salmon', linestyle=':', label=f'No-Reasoning Acc (Transcribed) ({bench_nr_trn_acc:.2f}%)')
#     ax.axhline(y=bench_bl_def_acc, color='green', linestyle='--', label=f'Baseline Acc (Original) ({bench_bl_def_acc:.2f}%)')
#     ax.axhline(y=bench_bl_trn_acc, color='lime', linestyle='--', label=f'Baseline Acc (Transcribed) ({bench_bl_trn_acc:.2f}%)')

#     # Formatting and Saving
#     base_title = f'Adding Mistakes Comparison: Original vs. Transcribed Audio ({dataset_name.upper()})'
#     if plot_group_name == 'aggregated':
#         subtitle = f'(Aggregated Across {num_chains} Common Chains)'
#     else:
#         subtitle = f'(For CoTs of Length {plot_group_name}, N={num_chains} Common Chains)'
#     ax.set_title(f"{base_title}\n{subtitle}", fontsize=16, pad=20)
#     ax.set_xlabel('% of Reasoning Chain Before Mistake', fontsize=12)
#     ax.set_ylabel('Rate (%)', fontsize=12)
#     ax.set_xlim(-5, 95); ax.set_ylim(0, 105); ax.legend(title='Condition & Metric', loc='best'); fig.tight_layout()

#     output_plot_dir = os.path.join(plots_dir, 'comparative_transcribed_audio', 'adding_mistakes', dataset_name, 'aggregated' if plot_group_name == 'aggregated' else 'grouped')
#     os.makedirs(output_plot_dir, exist_ok=True)
#     plot_path = os.path.join(output_plot_dir, f"compare_adding_mistakes_{dataset_name}_{plot_group_name}.png")
#     plt.savefig(plot_path, dpi=300); plt.close()
#     print(f"  - Plot saved successfully to: {plot_path}")


# def create_comparative_analysis(dataset_name: str, results_dir: str, plots_dir: str, generate_grouped: bool):
#     """ Main function to orchestrate the comparative analysis. """
#     print(f"\n--- Generating Comparative Adding Mistakes Analysis for: {dataset_name.upper()} ---")
    
#     try:
#         # Load all 6 required data files using our robust, condition-aware utility.
#         mistakes_default_df = load_results(results_dir, 'adding_mistakes', dataset_name, 'default')
#         mistakes_transcribed_df = load_results(results_dir, 'adding_mistakes', dataset_name, 'transcribed_audio')
#         baseline_default_df = load_results(results_dir, 'baseline', dataset_name, 'default')
#         baseline_transcribed_df = load_results(results_dir, 'baseline', dataset_name, 'transcribed_audio')
#         no_reasoning_default_df = load_results(results_dir, 'no_reasoning', dataset_name, 'default')
#         no_reasoning_transcribed_df = load_results(results_dir, 'no_reasoning', dataset_name, 'transcribed_audio')
#     except FileNotFoundError:
#         return

#     # --- Data Preparation ---
#     # Apply the "Meaningful Manipulation" filter to both dataframes.
#     mistakes_default_df = mistakes_default_df[mistakes_default_df['total_sentences_in_chain'] > 0].copy()
#     mistakes_transcribed_df = mistakes_transcribed_df[mistakes_transcribed_df['total_sentences_in_chain'] > 0].copy()
    
#     # Calculate the x-axis variable for both.
#     mistakes_default_df['percent_before_mistake'] = ((mistakes_default_df['mistake_position'] - 1) / mistakes_default_df['total_sentences_in_chain']) * 100
#     mistakes_transcribed_df['percent_before_mistake'] = ((mistakes_transcribed_df['mistake_position'] - 1) / mistakes_transcribed_df['total_sentences_in_chain']) * 100

#     print("Generating main aggregated plot...")
#     plot_comparative_graph(mistakes_default_df, mistakes_transcribed_df, baseline_default_df, baseline_transcribed_df, no_reasoning_default_df, no_reasoning_transcribed_df, 'aggregated', dataset_name, plots_dir)

#     if generate_grouped:
#         print("\nGenerating per-length grouped plots...")
#         grouped_default = mistakes_default_df.groupby('total_sentences_in_chain')
#         grouped_transcribed = mistakes_transcribed_df.groupby('total_sentences_in_chain')
#         common_lengths = set(grouped_default.groups.keys()) & set(grouped_transcribed.groups.keys())

#         for total_steps in sorted(list(common_lengths)):
#             group_df_default = grouped_default.get_group(total_steps)
#             group_df_transcribed = grouped_transcribed.get_group(total_steps)
            
#             if len(group_df_default[['id', 'chain_id']].drop_duplicates()) > 10 and \
#                len(group_df_transcribed[['id', 'chain_id']].drop_duplicates()) > 10:
#                 plot_comparative_graph(group_df_default, group_df_transcribed, baseline_default_df, baseline_transcribed_df, no_reasoning_default_df, no_reasoning_transcribed_df, f'{total_steps}_sentences', dataset_name, plots_dir)
#             else:
#                 print(f"  - Skipping plot for CoTs of length {total_steps} due to insufficient data.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate comparative Adding Mistakes plots.")
#     parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to analyze (e.g., 'mmar' or 'all').")
#     parser.add_argument('--results_dir', type=str, default='./results')
#     parser.add_argument('--plots_dir', type=str, default='./plots')
#     parser.add_argument('--grouped', action='store_true', help='Generate detailed plots for each CoT length.')
#     args = parser.parse_args()
    
#     if args.dataset == 'all':
#         try:
#             default_dir = os.path.join(args.results_dir, 'adding_mistakes')
#             datasets_default = set([f.replace('adding_mistakes_', '').replace('_default.jsonl', '') for f in os.listdir(default_dir) if f.endswith('_default.jsonl')])
#             datasets_transcribed = set([f.replace('adding_mistakes_', '').replace('_transcribed_audio.jsonl', '') for f in os.listdir(default_dir) if f.endswith('_transcribed_audio.jsonl')])
#             common_datasets = sorted(list(datasets_default & datasets_transcribed))
#             print(f"Found common datasets for comparison: {common_datasets}")
#             for dataset in common_datasets:
#                 create_comparative_analysis(dataset, args.results_dir, args.plots_dir, args.grouped)
#         except FileNotFoundError:
#             print(f"Could not find 'adding_mistakes' results directory at {default_dir}.")
#     else:
#         create_comparative_analysis(args.dataset, args.results_dir, args.plots_dir, args.grouped)