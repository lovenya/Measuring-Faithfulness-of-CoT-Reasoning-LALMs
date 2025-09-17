# data_processing/merge_parallel_results.py

"""
This is a standalone utility script that acts as the "GATHER" step in our
parallel processing workflow.

Its purpose is to find all the separate '.part_N.jsonl' output files generated
by a Slurm job array and merge them into a single, consolidated, and final
results file.

This script is designed to be run once, after all the parallel jobs for a
specific experiment have completed successfully.
"""

import os
import argparse
import glob

def merge_part_files(model: str, experiment: str, dataset: str, results_dir: str, restricted: bool):
    """
    Finds all .part_*.jsonl files for a given experiment run, concatenates them,
    and saves them to a final, consolidated .jsonl file.
    """
    print(f"\n--- Merging Parallel Results ---")
    print(f"  - Model: {model.upper()}")
    print(f"  - Experiment: {experiment.upper()}")
    print(f"  - Dataset: {dataset.upper()}")
    print(f"  - Run Mode: {'RESTRICTED' if restricted else 'FULL DATASET'}")

    # --- 1. Define the Search Directory and Pattern ---
    # We construct the path to the directory where the part files are located.
    # e.g., 'results/salmonn/adding_mistakes/'
    search_dir = os.path.join(results_dir, model, experiment)
    
    # We construct the base filename to search for.
    # e.g., 'adding_mistakes_salmonn_mmar-restricted'
    base_filename = f"{experiment}_{model}_{dataset}"
    if restricted:
        base_filename += "-restricted"
        
    # We create a glob pattern to find all files that match our base name
    # and have the '.part_*.jsonl' suffix.
    search_pattern = os.path.join(search_dir, f"{base_filename}.part_*.jsonl")
    
    print(f"\nSearching for part files in: '{search_dir}'")
    print(f"Using search pattern: '{os.path.basename(search_pattern)}'")

    # --- 2. Find and Validate the Part Files ---
    # glob.glob finds all files that match the pattern. We sort them to be safe.
    part_files = sorted(glob.glob(search_pattern))

    if not part_files:
        print("\nFATAL: No part files found to merge. Please check the model, experiment, and dataset names.")
        print(f"         Searched for: {search_pattern}")
        return

    print(f"\nFound {len(part_files)} part files to merge:")
    for path in part_files:
        print(f"  - {os.path.basename(path)}")

    # --- 3. Define the Final Output Path ---
    # The final output file is the base filename without the '.part_N' suffix.
    final_output_filename = f"{base_filename}.jsonl"
    final_output_path = os.path.join(search_dir, final_output_filename)

    # --- 4. Merge the Files ---
    print(f"\nMerging content into final output file: '{final_output_path}'...")
    total_lines_written = 0
    with open(final_output_path, 'w') as f_out:
        for part_file in part_files:
            with open(part_file, 'r') as f_in:
                for line in f_in:
                    # We simply copy each line from the part file to the final file.
                    f_out.write(line)
                    total_lines_written += 1
    
    print(f"\n--- Merge complete. ---")
    print(f"  - Total lines from all parts: {total_lines_written}")
    print(f"  - Final consolidated file saved to: {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge the parallel output files from a Slurm job array into a single results file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="The model alias (e.g., 'salmonn').")
    parser.add_argument('--experiment', type=str, required=True, help="The experiment name (e.g., 'adding_mistakes').")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset alias (e.g., 'mmar').")
    parser.add_argument('--results_dir', type=str, default='./results', help="Path to the main results directory.")
    parser.add_argument('--restricted', action='store_true', help="Specify if the run was on the '-restricted' subset.")
    args = parser.parse_args()

    merge_part_files(args.model, args.experiment, args.dataset, args.results_dir, args.restricted)