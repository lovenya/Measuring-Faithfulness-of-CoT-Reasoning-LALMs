#!/usr/bin/env python3
# data_processing/merge_combined_parallel_results.py

"""
GATHER step for parallel processing of combined experiment results.

This script merges the output files from parallel array jobs back into
a single consolidated results file.

The output files from main.py with --use-external-perturbations will have
the "-mistral" suffix in their names.

Usage:
    python data_processing/merge_combined_parallel_results.py \\
        --model qwen \\
        --dataset mmar \\
        --experiment adding_mistakes

    # Or merge all results:
    python data_processing/merge_combined_parallel_results.py --all
"""

import os
import argparse
import glob

# Constants
MODELS = ["qwen", "salmonn"]
DATASETS = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]
EXPERIMENTS = ["adding_mistakes", "paraphrasing"]
RESULTS_DIR = "results"


def merge_part_files(model: str, dataset: str, experiment: str) -> bool:
    """
    Merge part files for a specific model/dataset/experiment combination.
    
    Returns True if successful, False otherwise.
    """
    # Output files from combined runs have "-mistral" suffix
    # e.g., adding_mistakes_qwen_mmar-restricted-mistral.part_1.jsonl
    
    search_dir = os.path.join(RESULTS_DIR, model, experiment)
    base_filename = f"{experiment}_{model}_{dataset}-restricted-mistral"
    
    search_pattern = os.path.join(search_dir, f"{base_filename}.part_*.jsonl")
    
    print(f"\n--- Merging: {model}/{dataset}/{experiment} ---")
    print(f"  Search dir: {search_dir}")
    print(f"  Pattern: {base_filename}.part_*.jsonl")
    
    # Find all part files
    part_files = sorted(glob.glob(search_pattern))
    
    if not part_files:
        print(f"  ⚠️ SKIP: No part files found")
        return False
    
    print(f"  Found {len(part_files)} part files")
    
    # Define output path
    final_output_path = os.path.join(search_dir, f"{base_filename}.jsonl")
    
    # Merge files
    total_lines = 0
    with open(final_output_path, 'w') as f_out:
        for part_file in part_files:
            with open(part_file, 'r') as f_in:
                for line in f_in:
                    f_out.write(line)
                    total_lines += 1
    
    print(f"  ✓ Merged {total_lines} lines -> {os.path.basename(final_output_path)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge parallel output files into single results file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, choices=MODELS, help="Model name")
    parser.add_argument('--dataset', type=str, choices=DATASETS, help="Dataset name")
    parser.add_argument('--experiment', type=str, choices=EXPERIMENTS, help="Experiment type")
    parser.add_argument('--all', action='store_true', help="Merge all results")
    
    args = parser.parse_args()
    
    if args.all:
        print(f"\n=== Merging ALL parallel results ===")
        success = 0
        skipped = 0
        
        for model in MODELS:
            for dataset in DATASETS:
                for experiment in EXPERIMENTS:
                    if merge_part_files(model, dataset, experiment):
                        success += 1
                    else:
                        skipped += 1
        
        print(f"\n=== SUMMARY ===")
        print(f"  Successful: {success}")
        print(f"  Skipped: {skipped}")
    else:
        if not args.model or not args.dataset or not args.experiment:
            parser.error("--model, --dataset, and --experiment are required unless --all is specified")
        merge_part_files(args.model, args.dataset, args.experiment)


if __name__ == "__main__":
    main()
