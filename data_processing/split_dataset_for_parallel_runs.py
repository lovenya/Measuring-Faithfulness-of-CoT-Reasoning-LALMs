# data_processing/split_dataset_for_parallel_runs.py

"""
This is a standalone utility script that acts as the "SCATTER" step in our
parallel processing workflow.

Its purpose is to take our large, pre-filtered '-restricted.jsonl' files for
both 'baseline' and 'no_reasoning' experiments and split them into a specified
number of smaller, numbered chunk files. This prepares the data to be processed
in parallel by a Slurm job array.

The splitting is done intelligently and synchronously. The 'baseline' file is
treated as the "master." We first determine how to split the question IDs from
the baseline file, and then we apply that exact same split to the 'no_reasoning'
file. This guarantees that for any given part number N, the baseline part file
and the no_reasoning part file will contain the exact same set of questions.
"""

import os
import json
import argparse
import collections
import numpy as np

def split_restricted_files(model: str, dataset: str, num_parts: int, results_dir: str):
    """
    Reads restricted baseline and no_reasoning files, splits their content
    synchronously by question ID, and writes out new '.part_N.jsonl' files for both.
    """
    print(f"\n--- Splitting Restricted Datasets for Parallel Runs ---")
    print(f"  - Model: {model.upper()}")
    print(f"  - Dataset: {dataset.upper()}")
    print(f"  - Number of Parts: {num_parts}")

    # --- 1. Define Input Paths ---
    baseline_input_path = os.path.join(results_dir, model, 'baseline', f'baseline_{model}_{dataset}-restricted.jsonl')
    no_reasoning_input_path = os.path.join(results_dir, model, 'no_reasoning', f'no_reasoning_{model}_{dataset}-restricted.jsonl')

    for path in [baseline_input_path, no_reasoning_input_path]:
        if not os.path.exists(path):
            print(f"\nFATAL: A required '-restricted' input file was not found.")
            print(f"Looked for: '{path}'")
            print("Please run 'create_restricted_dataset.py' first for this model and dataset.")
            return

    # --- 2. Create the "Master Split Plan" from the Baseline File ---
    print(f"\nReading and grouping data from MASTER file: '{baseline_input_path}'...")
    baseline_chains_by_id = collections.defaultdict(list)
    with open(baseline_input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            baseline_chains_by_id[data['id']].append(data)
    
    unique_question_ids = list(baseline_chains_by_id.keys())
    print(f"Found {len(unique_question_ids)} unique question IDs in the master file.")

    if len(unique_question_ids) < num_parts:
        print(f"\nWARNING: The number of unique questions ({len(unique_question_ids)}) is less than the desired number of parts ({num_parts}).")
        print("         This will result in some empty part files. Consider a smaller number of parts.")

    # We split the list of unique IDs into N chunks. This is our "Master Plan".
    id_chunks = np.array_split(unique_question_ids, num_parts)

    # --- 3. Write the Baseline Part Files (Master) ---
    print("\nWriting new BASELINE part files...")
    for i, id_chunk in enumerate(id_chunks):
        part_num = i + 1
        output_path = baseline_input_path.replace('.jsonl', f'.part_{part_num}.jsonl')
        
        part_chains = [chain for q_id in id_chunk for chain in baseline_chains_by_id[q_id]]
            
        with open(output_path, 'w') as f_out:
            for chain in part_chains:
                f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")
        
        print(f"  - Part {part_num}/{num_parts}: Wrote {len(part_chains)} chains to '{output_path}'")

    # --- 4. Write the No-Reasoning Part Files (Slave) ---
    print(f"\nReading and grouping data from SLAVE file: '{no_reasoning_input_path}'...")
    no_reasoning_chains_by_id = collections.defaultdict(list)
    with open(no_reasoning_input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            no_reasoning_chains_by_id[data['id']].append(data)

    print("\nWriting new NO-REASONING part files (synchronized with baseline)...")
    for i, id_chunk in enumerate(id_chunks):
        part_num = i + 1
        output_path = no_reasoning_input_path.replace('.jsonl', f'.part_{part_num}.jsonl')
        
        # We use the exact same id_chunk from the Master Plan to select the data.
        # This guarantees a perfect 1-to-1 correspondence of questions.
        part_chains = [chain for q_id in id_chunk for chain in no_reasoning_chains_by_id[q_id]]
            
        with open(output_path, 'w') as f_out:
            for chain in part_chains:
                f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")
        
        print(f"  - Part {part_num}/{num_parts}: Wrote {len(part_chains)} chains to '{output_path}'")

    print("\n--- Synchronized splitting complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split restricted datasets into multiple parts for parallel processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="The model alias (e.g., 'salmonn').")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset alias (e.g., 'mmar').")
    parser.add_argument('--num-parts', type=int, required=True, help="The number of parallel chunks to create.")
    parser.add_argument('--results_dir', type=str, default='./results', help="Path to the main results directory.")
    args = parser.parse_args()

    split_restricted_files(args.model, args.dataset, args.num_parts, args.results_dir)