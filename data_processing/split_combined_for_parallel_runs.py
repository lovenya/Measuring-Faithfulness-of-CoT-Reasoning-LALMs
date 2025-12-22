#!/usr/bin/env python3
# data_processing/split_combined_for_parallel_runs.py

"""
SCATTER step for parallel processing of combined (baseline + Mistral perturbation) files.

This script takes the combined JSONL files from results/combined/ and splits them
into N parts for parallel processing via Slurm array jobs.

Unlike the original split script that splits by question ID, this splits the combined
files directly by trial (each line is one trial to run).

Usage:
    python data_processing/split_combined_for_parallel_runs.py \\
        --model qwen \\
        --dataset mmar \\
        --experiment adding_mistakes \\
        --num-parts 20

    # Or split all combined files:
    python data_processing/split_combined_for_parallel_runs.py --all --num-parts 20
"""

import os
import json
import argparse
import numpy as np

# Constants
MODELS = ["qwen", "salmonn"]
DATASETS = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]
EXPERIMENTS = ["adding_mistakes", "paraphrasing"]
COMBINED_DIR = "results/combined"


def get_combined_path(model: str, dataset: str, experiment: str) -> str:
    """Get path to combined file."""
    return os.path.join(COMBINED_DIR, f"{model}_{dataset}-restricted_{experiment}_combined.jsonl")


def split_combined_file(model: str, dataset: str, experiment: str, num_parts: int) -> bool:
    """
    Split a single combined file into N parts.
    
    Returns True if successful, False otherwise.
    """
    input_path = get_combined_path(model, dataset, experiment)
    
    if not os.path.exists(input_path):
        print(f"  ⚠️ SKIP: Combined file not found: {input_path}")
        return False
    
    print(f"\n--- Splitting: {model}/{dataset}/{experiment} ---")
    print(f"  Input: {input_path}")
    
    # Load all trials
    trials = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"  Loaded {len(trials)} trials")
    
    if len(trials) < num_parts:
        print(f"  ⚠️ WARNING: Only {len(trials)} trials, but {num_parts} parts requested.")
        print(f"              Some parts will be empty. Consider fewer parts.")
    
    # Split into N chunks
    trial_indices = list(range(len(trials)))
    chunks = np.array_split(trial_indices, num_parts)
    
    # Write part files
    for i, chunk_indices in enumerate(chunks):
        part_num = i + 1
        output_path = input_path.replace('.jsonl', f'.part_{part_num}.jsonl')
        
        chunk_trials = [trials[idx] for idx in chunk_indices]
        
        with open(output_path, 'w') as f_out:
            for trial in chunk_trials:
                f_out.write(json.dumps(trial, ensure_ascii=False) + "\n")
        
        print(f"    Part {part_num:2d}/{num_parts}: {len(chunk_trials):5d} trials -> {os.path.basename(output_path)}")
    
    print(f"  ✓ Split complete: {num_parts} parts created")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split combined files into parts for parallel processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, choices=MODELS, help="Model name")
    parser.add_argument('--dataset', type=str, choices=DATASETS, help="Dataset name")
    parser.add_argument('--experiment', type=str, choices=EXPERIMENTS, help="Experiment type")
    parser.add_argument('--num-parts', type=int, required=True, help="Number of parts to split into")
    parser.add_argument('--all', action='store_true', help="Split all combined files")
    
    args = parser.parse_args()
    
    if args.all:
        print(f"\n=== Splitting ALL combined files into {args.num_parts} parts ===")
        success = 0
        skipped = 0
        
        for model in MODELS:
            for dataset in DATASETS:
                for experiment in EXPERIMENTS:
                    if split_combined_file(model, dataset, experiment, args.num_parts):
                        success += 1
                    else:
                        skipped += 1
        
        print(f"\n=== SUMMARY ===")
        print(f"  Successful: {success}")
        print(f"  Skipped: {skipped}")
    else:
        if not args.model or not args.dataset or not args.experiment:
            parser.error("--model, --dataset, and --experiment are required unless --all is specified")
        split_combined_file(args.model, args.dataset, args.experiment, args.num_parts)


if __name__ == "__main__":
    main()
