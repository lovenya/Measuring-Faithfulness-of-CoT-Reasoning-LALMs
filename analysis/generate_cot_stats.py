# analysis/generate_cot_stats.py

import os
import pandas as pd
import argparse
from collections import Counter
from utils import load_results

def create_analysis(model_name: str, dataset_name: str, results_dir: str, num_samples: int, num_chains: int):
    """
    Loads data and generates statistics on the distribution of CoT lengths.
    """
    print(f"\n--- CoT Statistics for: {model_name.upper()} on {dataset_name.upper()} ---")
    
    try:
        baseline_df = load_results(model_name, results_dir, 'baseline', dataset_name)
        early_df = load_results(model_name, results_dir, 'early_answering', dataset_name)
    except FileNotFoundError:
        return

    # --- Data Filtering based on CLI arguments ---
    if num_samples is not None and num_samples > 0:
        unique_ids = baseline_df['id'].unique()[:num_samples]
        early_df = early_df[early_df['id'].isin(unique_ids)]

    # --- THE CRITICAL FIX ---
    # Now correctly filters the DataFrame to include only the first N chains per sample.
    if num_chains is not None and num_chains > 0:
        early_df = early_df[early_df['chain_id'] < num_chains]
    # --- END OF FIX ---

    if early_df.empty:
        print("  - No data found after filtering. Skipping stats generation.")
        return

    # --- Core Logic ---
    unique_chains_df = early_df.drop_duplicates(subset=['id', 'chain_id'])

    if unique_chains_df.empty:
        print("  - No valid chains to analyze.")
        return

    total_unique_chains = len(unique_chains_df)
    total_unique_questions = len(unique_chains_df['id'].unique())
    sentence_counts = Counter(unique_chains_df['total_sentences_in_chain'])
    
    # --- Print Formatted Report ---
    print(f"\nOverall Stats:")
    print(f"  - Total Unique Questions Analyzed: {total_unique_questions}")
    print(f"  - Total Unique Chains Analyzed:  {total_unique_chains}")

    print(f"\nDistribution of CoT Lengths (by sentence count):")
    for length in sorted(sentence_counts.keys()):
        count = sentence_counts[length]
        print(f"  - {length:2d} sentences: {count:5d} chains")

    summary_stats = unique_chains_df['total_sentences_in_chain'].describe()
    print(f"\nSummary Statistics:")
    print(f"  - Mean:   {summary_stats['mean']:.2f} sentences")
    print(f"  - Std Dev: {summary_stats['std']:.2f}")
    print(f"  - Min:    {int(summary_stats['min'])} sentences")
    print(f"  - 25%:    {int(summary_stats['25%'])} sentences")
    print(f"  - Median: {int(summary_stats['50%'])} sentences")
    print(f"  - 75%:    {int(summary_stats['75%'])} sentences")
    print(f"  - Max:    {int(summary_stats['max'])} sentences")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate statistics on CoT lengths from LALM results.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--num-chains', type=int, default=None)
    args = parser.parse_args()
    
    if args.dataset == 'all':
        try:
            exp_dir = os.path.join(args.results_dir, args.model, 'early_answering')
            dataset_names = sorted([f.replace(f'early_answering_{args.model}_', '').replace('.jsonl', '') for f in os.listdir(exp_dir) if f.endswith('.jsonl')])
            print(f"Found datasets for model '{args.model}': {dataset_names}")
            for dataset in dataset_names:
                create_analysis(args.model, dataset, args.results_dir, args.num_samples, args.num_chains)
        except FileNotFoundError:
            print(f"Could not find early_answering directory for model '{args.model}' at {exp_dir}.")
    else:
        create_analysis(args.model, args.dataset, args.results_dir, args.num_samples, args.num_chains)