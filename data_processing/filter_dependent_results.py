# data_processing/filter_dependent_results.py

"""
This is a standalone utility script for post-processing our experimental results.

Its purpose is to create a "restricted" version of a dependent experiment's
results (e.g., adding_mistakes) based on a pre-existing restricted baseline file.

WHY THIS IS NEEDED:
We decided to standardize our analysis on a specific subset of data (e.g., the
first 3 chains per question, with 1-6 sentence CoTs). We have already run a
script to create 'baseline...-restricted.jsonl' and 'no_reasoning...-restricted.jsonl'.
However, some dependent experiments (like for the 'qwen' model) were already run on the
FULL baseline data. This script "corrects" those full results by filtering them
down to only include the chains present in the restricted baseline file, ensuring
perfect consistency for our final analysis.
"""

import os
import json
import argparse

def filter_one_experiment(model: str, dataset: str, experiment: str, results_dir: str):
    """
    Filters a single dependent experiment's results file based on the
    corresponding restricted baseline file.
    """
    print(f"\n--- Filtering Dependent Results ---")
    print(f"  - Model:      {model.upper()}")
    print(f"  - Dataset:    {dataset.upper()}")
    print(f"  - Experiment: {experiment.upper()}")

    # --- 1. Define Paths ---
    # This is our "source of truth". It contains the exact (id, chain_id) pairs we want to keep.
    restricted_baseline_path = os.path.join(results_dir, model, 'baseline', f'baseline_{model}_{dataset}-restricted.jsonl')
    
    # This is the full, original results file that we need to filter.
    full_dependent_input_path = os.path.join(results_dir, model, experiment, f'{experiment}_{model}_{dataset}.jsonl')
    
    # This is the new, filtered file we will create.
    restricted_dependent_output_path = full_dependent_input_path.replace('.jsonl', '-restricted.jsonl')

    # --- 2. Safety Checks ---
    for path in [restricted_baseline_path, full_dependent_input_path]:
        if not os.path.exists(path):
            print(f"  - FATAL: Required input file not found at '{path}'. Cannot proceed.")
            return

    # --- 3. Build the Set of Valid Chains ---
    print(f"Reading the list of valid chains from: {restricted_baseline_path}")
    valid_chains = set()
    with open(restricted_baseline_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # We create a set of tuples for highly efficient lookup.
            valid_chains.add((data['id'], data['chain_id']))
    print(f"  - Found {len(valid_chains)} unique chains to keep.")

    # --- 4. Filter the Dependent Results File ---
    print(f"Filtering the full results file: {full_dependent_input_path}")
    filtered_results = []
    with open(full_dependent_input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # This is the core logic: we only keep a result if its (id, chain_id)
            # pair exists in our set of valid chains.
            if (data['id'], data['chain_id']) in valid_chains:
                filtered_results.append(data)

    # --- 5. Write the New Restricted File ---
    with open(restricted_dependent_output_path, 'w') as f:
        for result in filtered_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"  - Successfully wrote {len(filtered_results)} filtered results to: {restricted_dependent_output_path}")
    print("--- Filtering complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter existing dependent experiment results to match a restricted baseline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="The model alias (e.g., 'qwen', 'salmonn').")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset alias (e.g., 'mmar', or 'all').")
    parser.add_argument('--experiment', type=str, required=True, help="The dependent experiment to filter (e.g., 'adding_mistakes').")
    parser.add_argument('--results_dir', type=str, default='./results', help="Path to the main results directory.")
    args = parser.parse_args()

    if args.dataset == 'all':
        try:
            model_baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            dataset_names = sorted(list(set([
                f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '').replace('-restricted', '')
                for f in os.listdir(model_baseline_dir) if f.endswith('.jsonl') and not f.endswith('-restricted.jsonl')
            ])))
            print(f"Found {len(dataset_names)} datasets for model '{args.model}': {dataset_names}")
            for dataset_name in dataset_names:
                filter_one_experiment(args.model, dataset_name, args.experiment, args.results_dir)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at '{model_baseline_dir}'.")
    else:
        filter_one_experiment(args.model, args.dataset, args.experiment, args.results_dir)
