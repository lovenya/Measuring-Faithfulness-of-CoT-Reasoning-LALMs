# data_processing/create_restricted_dataset.py

"""
This is a standalone utility script for pre-processing our experimental results.

This upgraded version now performs a sophisticated, two-stage filtering process:
1.  It first filters by the number of chains per question, allowing us to
    standardize our analysis across models (e.g., using only the first 3 chains).
2.  It then filters that subset by CoT sentence length (e.g., keeping only
    chains that are 1-6 sentences long).

This is crucial for creating fair, apples-to-apples comparisons in our final analysis.
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


def create_restricted_files(model: str, dataset: str, results_dir: str, num_chains: int | None, skip_no_reasoning: bool = False):
    """
    Reads full foundational results and writes new '-restricted.jsonl' versions
    based on the specified number of chains and sentence lengths.
    """
    print(f"\n--- Creating Restricted Dataset for Model: {model.upper()}, Dataset: {dataset.upper()} ---")
    if num_chains:
        print(f"  - Restricting to the first {num_chains} chains per question.")
    if skip_no_reasoning:
        print(f"  - Skipping no_reasoning file (baseline-only mode).")

    # --- 1. Define Input and Output Paths ---
    # This logic correctly constructs the paths to the full results files.
    baseline_input_path = os.path.join(results_dir, model, 'baseline', f'baseline_{model}_{dataset}.jsonl')
    no_reasoning_input_path = os.path.join(results_dir, model, 'no_reasoning', f'no_reasoning_{model}_{dataset}.jsonl')

    # The output filename is now standardized to '-restricted.jsonl' and does NOT
    # include the number of chains. This ensures compatibility with all other scripts.
    baseline_output_path = baseline_input_path.replace('.jsonl', '-restricted.jsonl')
    no_reasoning_output_path = no_reasoning_input_path.replace('.jsonl', '-restricted.jsonl')

    # Check if the required input files exist.
    if not os.path.exists(baseline_input_path):
        print(f"  - FATAL: Required input file not found at '{baseline_input_path}'. Cannot proceed.")
        return
    
    if not skip_no_reasoning and not os.path.exists(no_reasoning_input_path):
        print(f"  - FATAL: Required input file not found at '{no_reasoning_input_path}'.")
        print(f"    Use --skip-no-reasoning to create restricted baseline without no_reasoning file.")
        return

    # --- 2. Stage 1: Filter by Number of Chains ---
    print(f"Reading full baseline data from: {baseline_input_path}")
    
    # We first group all chains by their question ID.
    chains_by_question = collections.defaultdict(list)
    with open(baseline_input_path, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            chains_by_question[data['id']].append(data)

    # Now, we create our initial subset based on the num_chains parameter.
    chain_filtered_subset = []
    for q_id, chains in chains_by_question.items():
        # Sort chains by chain_id to ensure we always take the first N.
        chains.sort(key=lambda x: x['chain_id'])
        if num_chains:
            # If a limit is specified, take only up to that many chains.
            chain_filtered_subset.extend(chains[:num_chains])
        else:
            # If no limit, take all chains.
            chain_filtered_subset.extend(chains)
    
    print(f"  - Stage 1: Selected {len(chain_filtered_subset)} chains based on --num-chains limit.")

    # --- 3. Stage 2: Filter by Sentence Length ---
    valid_lengths = {1, 2, 3, 4, 5, 6}
    restricted_baseline_chains = []
    valid_question_ids = set()

    for chain_data in chain_filtered_subset:
        num_sentences = len(nltk.sent_tokenize(chain_data['sanitized_cot']))
        if num_sentences in valid_lengths:
            restricted_baseline_chains.append(chain_data)
            valid_question_ids.add(chain_data['id'])

    print(f"  - Stage 2: Filtered down to {len(restricted_baseline_chains)} chains with valid sentence lengths (1-6).")

    # Write the final restricted baseline file.
    with open(baseline_output_path, 'w') as f_out:
        for chain in restricted_baseline_chains:
            f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")
    print(f"  - Wrote {len(restricted_baseline_chains)} valid chains to: {baseline_output_path}")

    # --- 4. Filter the No-Reasoning File based on the final set of valid IDs ---
    if skip_no_reasoning:
        print(f"  - Skipping no-reasoning filtering (--skip-no-reasoning enabled).")
    else:
        print(f"Reading full no-reasoning data from: {no_reasoning_input_path}")
        restricted_no_reasoning_chains = []
        with open(no_reasoning_input_path, 'r') as f_in:
            for line in f_in:
                data = json.loads(line)
                # We only keep data for questions that are present in our final restricted baseline file.
                # We also respect the num_chains limit for this file.
                if data['id'] in valid_question_ids and (not num_chains or data['chain_id'] < num_chains):
                    restricted_no_reasoning_chains.append(data)

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
    
    parser.add_argument(
        '--num-chains', 
        type=int, 
        default=None, 
        help="If specified, restrict the dataset to the first N chains for each question."
    )
    parser.add_argument(
        '--skip-no-reasoning',
        action='store_true',
        help="Skip no_reasoning file filtering. Use when only baseline is needed for dependent experiments."
    )
    
    args = parser.parse_args()

    if args.dataset == 'all':
        try:
            model_baseline_dir = os.path.join(args.results_dir, args.model, 'baseline')
            # This logic extracts the dataset names from the filenames, safely ignoring suffixes.
            dataset_names = sorted(list(set([
                f.replace(f'baseline_{args.model}_', '').replace('.jsonl', '').replace('-restricted', '')
                for f in os.listdir(model_baseline_dir) if f.endswith('.jsonl')
            ])))
            print(f"Found {len(dataset_names)} datasets for model '{args.model}': {dataset_names}")
            for dataset_name in dataset_names:
                create_restricted_files(args.model, dataset_name, args.results_dir, args.num_chains, args.skip_no_reasoning)
        except FileNotFoundError:
            print(f"Could not find baseline directory for model '{args.model}' at '{model_baseline_dir}'.")
    else:
        create_restricted_files(args.model, args.dataset, args.results_dir, args.num_chains, args.skip_no_reasoning)