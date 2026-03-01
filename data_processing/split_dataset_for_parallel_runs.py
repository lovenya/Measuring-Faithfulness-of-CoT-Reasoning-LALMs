# data_processing/split_dataset_for_parallel_runs.py

"""
Standalone utility script: the "SCATTER" step in the parallel processing workflow.

Takes a baseline JSONL file and splits it into N smaller numbered chunk files,
preparing the data for parallel processing via Slurm job arrays.

The splitting is done by question ID: we first collect all unique question IDs,
split them into N groups, and write the trials for each group to its own part file.

Path building mirrors main.py (--model + --dataset), with --restricted as opt-in.
"""

import os
import json
import argparse
import collections
import numpy as np


def build_baseline_path(model: str, dataset: str, results_dir: str, restricted: bool) -> str:
    """Build the input baseline file path from CLI args, matching main.py conventions."""
    filename = f"baseline_{model}_{dataset}"
    if restricted:
        filename += "-restricted"
    filename += ".jsonl"
    return os.path.join(results_dir, model, "baseline", filename)


def build_no_reasoning_path(model: str, dataset: str, results_dir: str, restricted: bool) -> str:
    """Build the input no_reasoning file path from CLI args."""
    filename = f"no_reasoning_{model}_{dataset}"
    if restricted:
        filename += "-restricted"
    filename += ".jsonl"
    return os.path.join(results_dir, model, "no_reasoning", filename)


def split_file_by_question_id(input_path: str, num_parts: int, label: str):
    """
    Read a JSONL file, group trials by 'id', split the question IDs into
    N chunks, and write numbered part files next to the original.

    Returns the list of id_chunks (the "master plan") for synchronized splitting.
    """
    print(f"\nReading and grouping data from {label} file: '{input_path}'...")
    chains_by_id = collections.defaultdict(list)
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            chains_by_id[data['id']].append(data)

    unique_question_ids = list(chains_by_id.keys())
    print(f"Found {len(unique_question_ids)} unique question IDs.")

    if len(unique_question_ids) < num_parts:
        print(f"\nWARNING: Only {len(unique_question_ids)} questions but {num_parts} parts requested.")
        print("         Some parts will be empty. Consider fewer parts.")

    id_chunks = np.array_split(unique_question_ids, num_parts)

    print(f"\nWriting {label.upper()} part files...")
    for i, id_chunk in enumerate(id_chunks):
        part_num = i + 1
        output_path = input_path.replace('.jsonl', f'.part_{part_num}.jsonl')

        part_chains = [chain for q_id in id_chunk for chain in chains_by_id[q_id]]

        with open(output_path, 'w') as f_out:
            for chain in part_chains:
                f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")

        print(f"  - Part {part_num}/{num_parts}: {len(part_chains)} chains -> '{os.path.basename(output_path)}'")

    return id_chunks, chains_by_id


def split_with_master_plan(input_path: str, id_chunks, num_parts: int, label: str):
    """
    Split a secondary file using the same id_chunks from the master file.
    This guarantees 1-to-1 correspondence of questions across files.
    """
    print(f"\nReading data from {label} (SLAVE) file: '{input_path}'...")
    chains_by_id = collections.defaultdict(list)
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            chains_by_id[data['id']].append(data)

    print(f"Writing {label.upper()} part files (synchronized with master)...")
    for i, id_chunk in enumerate(id_chunks):
        part_num = i + 1
        output_path = input_path.replace('.jsonl', f'.part_{part_num}.jsonl')

        part_chains = [chain for q_id in id_chunk for chain in chains_by_id[q_id]]

        with open(output_path, 'w') as f_out:
            for chain in part_chains:
                f_out.write(json.dumps(chain, ensure_ascii=False) + "\n")

        print(f"  - Part {part_num}/{num_parts}: {len(part_chains)} chains -> '{os.path.basename(output_path)}'")


def split_restricted_files(model: str, dataset: str, num_parts: int,
                           results_dir: str, restricted: bool = False,
                           skip_no_reasoning: bool = False):
    """
    Main entry point. Builds paths from CLI args and splits baseline
    (and optionally no_reasoning) into N parts.
    """
    print(f"\n--- Splitting Datasets for Parallel Runs ---")
    print(f"  - Model: {model.upper()}")
    print(f"  - Dataset: {dataset.upper()}")
    print(f"  - Number of Parts: {num_parts}")
    print(f"  - Mode: {'RESTRICTED' if restricted else 'FULL DATASET'}")
    if skip_no_reasoning:
        print(f"  - Skipping no_reasoning splitting")

    # --- Build paths from CLI args (mirrors main.py) ---
    baseline_path = build_baseline_path(model, dataset, results_dir, restricted)

    if not os.path.exists(baseline_path):
        print(f"\nFATAL: Baseline file not found: '{baseline_path}'")
        return

    # --- Split baseline (MASTER) ---
    id_chunks, _ = split_file_by_question_id(baseline_path, num_parts, "baseline")

    # --- Split no_reasoning (SLAVE, synchronized) ---
    if not skip_no_reasoning:
        no_reasoning_path = build_no_reasoning_path(model, dataset, results_dir, restricted)
        if not os.path.exists(no_reasoning_path):
            print(f"\nFATAL: No-reasoning file not found: '{no_reasoning_path}'")
            print("Use --skip-no-reasoning if you only need baseline splits.")
            return
        split_with_master_plan(no_reasoning_path, id_chunks, num_parts, "no_reasoning")
    else:
        print("\nSkipping no-reasoning file splitting (--skip-no-reasoning enabled).")

    print("\n--- Splitting complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split baseline datasets into multiple parts for parallel processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="The model alias (e.g., 'flamingo_hf').")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset alias (e.g., 'mmar').")
    parser.add_argument('--num-parts', type=int, required=True, help="The number of parallel chunks to create.")
    parser.add_argument('--results-dir', type=str, default='./results', help="Path to the main results directory.")
    parser.add_argument('--restricted', action='store_true', help="Use the '-restricted' subset (opt-in, NOT default).")
    parser.add_argument('--skip-no-reasoning', action='store_true', help="Skip no_reasoning file splitting (baseline-only mode).")
    args = parser.parse_args()

    split_restricted_files(args.model, args.dataset, args.num_parts,
                           args.results_dir, args.restricted, args.skip_no_reasoning)