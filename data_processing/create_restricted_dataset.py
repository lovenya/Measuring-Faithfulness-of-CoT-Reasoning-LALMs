# data_processing/create_restricted_dataset.py

"""
This is a standalone utility script for pre-processing our experimental results.

The primary goal of this script is to create a "restricted" subset of our data.
Specifically, it filters our foundational experiment results (baseline and no_reasoning)
to include ONLY those reasoning chains that are 3, 4, 5, or 6 sentences long.

This is a crucial optimization step. It allows our most computationally expensive
dependent experiments (like adding_mistakes) to run only on the data that will
be relevant for our final analysis, saving a massive amount of time and compute.
"""

import os
import json
import argparse
import nltk
import collections

# --- Environment Setup for NLTK ---
# We must ensure NLTK knows where to find our offline 'punkt' package.
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print(f"FATAL: NLTK 'punkt' model not found in '{local_nltk_data_path}'.")
        exit(1)
else:
    print(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    exit(1)


def create_restricted_files(model: str, dataset: str, results_dir: str):
    """
    Reads full baseline and no_reasoning files and writes new '-restricted.jsonl'
    versions containing only data for 3, 4, 5, or 6-step CoTs.
    """
    print(f"\n--- Creating Restricted Dataset for Model: {model.upper()}, Dataset: {dataset.upper()} ---")

    # --- 1. Define Input and Output Paths ---
    # This logic correctly constructs the paths to the full results files.
    baseline_input_path = os.path.join(results_dir, model, 'baseline', f'baseline_{model}_{dataset}.jsonl')
    no_reasoning_input_path = os.path.join(results_dir, model, 'no_reasoning', f'no_reasoning_{model}_{dataset}.jsonl')

    # This logic creates the new '-restricted' filenames.
    baseline_output_path = baseline_input_path.replace('.jsonl', '-restricted.jsonl')
    no_reasoning_output_path = no_reasoning_input_path.replace('.jsonl', '-restricted.jsonl')

    # Check if the required input files exist.
    for path in [baseline_input_path, no_reasoning_input_path]:
        if not os.path.exists(path):
            print(f"  - FATAL: Required input file not found at '{path}'. Cannot proceed.")
            return

    # --- 2. Filter the Baseline File ---
    print(f"Reading full baseline data from: {baseline_input_path}")
    valid_lengths = {3, 4, 5, 6}
    restricted_baseline_chains = []
    valid_question_ids = set() # We'll track the IDs of questions that have at least one valid chain.

    with open(baseline_input_path, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            # We count the sentences in the 'sanitized_cot'.
            num_sentences = len(nltk.sent_tokenize(data['sanitized_cot']))
            
            if num_sentences in valid_lengths:
                # If the chain has a valid length, we keep it...
                restricted_baseline_chains.append(data)
                # ...and we record its question ID.
                valid_question_ids.add(data['id'])

    # Write the new restricted baseline file.
    with open(baseline_output_path, 'w') as f_out:
        for chain in restricted_baseline_chains:
            f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")
    
    print(f"  - Wrote {len(restricted_baseline_chains)} valid chains to: {baseline_output_path}")

    # --- 3. Filter the No-Reasoning File ---
    print(f"Reading full no-reasoning data from: {no_reasoning_input_path}")
    restricted_no_reasoning_chains = []
    with open(no_reasoning_input_path, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            # We only keep the no-reasoning data for the questions that are
            # present in our new restricted baseline file. This is crucial
            # for fair downstream analysis.
            if data['id'] in valid_question_ids:
                restricted_no_reasoning_chains.append(data)

    # Write the new restricted no-reasoning file.
    with open(no_reasoning_output_path, 'w') as f_out:
        for chain in restricted_no_reasoning_chains:
            f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")

    print(f"  - Wrote {len(restricted_no_reasoning_chains)} corresponding no-reasoning chains to: {no_reasoning_output_path}")
    print("--- Restricted dataset creation complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create restricted subsets of foundational experiment results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="The model alias (e.g., 'qwen', 'salmonn').")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset alias (e.g., 'mmar', or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results', help="Path to the main results directory.")
    args = parser.parse_args()

    if args.dataset == 'all':
        # If 'all' is specified, we find all datasets for the given model and process them.
        try:
            model_baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            # This logic extracts the dataset names from the filenames.
            dataset_names = sorted(list(set([
                f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '')
                for f in os.listdir(model_baseline_dir) if f.endswith('.jsonl')
            ])))
            print(f"Found {len(dataset_names)} datasets for model '{args.model}': {dataset_names}")
            for dataset_name in dataset_names:
                create_restricted_files(args.model, dataset_name, args.results_dir)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at '{model_baseline_dir}'.")
    else:
        create_restricted_files(args.model, args.dataset, args.results_dir)