# data_processing/add_chain_ids_to_qwen.py

"""
This is a one-time utility script to fix a data consistency issue in the
original Qwen 'no_reasoning' results.

The original script saved only one result per question, without a 'chain_id'.
This script reads those files, duplicates each entry 10 times, and adds a
'chain_id' from 0 to 9 to each new entry.

This ensures that the structure of the Qwen no_reasoning data perfectly
matches the structure of all other foundational results, which is critical
for the robustness of our downstream processing and analysis scripts.
"""

import os
import json
import collections

def upgrade_no_reasoning_file(filepath: str, num_chains: int = 10):
    """
    Reads a no_reasoning file, duplicates its entries, and overwrites it.
    """
    if not os.path.exists(filepath):
        print(f"  - File not found: {filepath}. Skipping.")
        return

    print(f"Processing file: {filepath}...")
    
    # We use a dictionary to ensure we only process unique question IDs,
    # just in case the original file had duplicates.
    original_entries = {}
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # We store the first entry we see for each ID.
                if data['id'] not in original_entries:
                    original_entries[data['id']] = data
            except (json.JSONDecodeError, KeyError):
                print(f"  - WARNING: Found and skipped a corrupted line.")
                continue

    # Now, we build the new, upgraded list of entries.
    upgraded_entries = []
    for q_id, original_data in original_entries.items():
        for i in range(num_chains):
            # Create a fresh copy for each new chain.
            new_entry = original_data.copy()
            # Add the crucial 'chain_id'.
            new_entry['chain_id'] = i
            upgraded_entries.append(new_entry)

    # Finally, we overwrite the original file with the new, uniform data.
    with open(filepath, 'w') as f:
        for entry in upgraded_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"  - Success! Upgraded {len(original_entries)} unique questions to {len(upgraded_entries)} total chains.")


def main():
    """
    Finds all 'no_reasoning' files for the 'qwen' model and upgrades them.
    """
    results_dir = './results'
    model_alias = 'qwen'
    experiment_name = 'no_reasoning'
    
    qwen_no_reasoning_dir = os.path.join(results_dir, model_alias, experiment_name)

    if not os.path.exists(qwen_no_reasoning_dir):
        print(f"FATAL: Directory not found: {qwen_no_reasoning_dir}")
        print("Please ensure your Qwen no_reasoning results are in the correct location.")
        return

    print(f"--- Upgrading Qwen 'no_reasoning' files in: {qwen_no_reasoning_dir} ---")
    
    # Find all the relevant JSONL files to upgrade.
    files_to_process = [
        os.path.join(qwen_no_reasoning_dir, f)
        for f in os.listdir(qwen_no_reasoning_dir)
        if f.endswith('.jsonl')
    ]

    if not files_to_process:
        print("No .jsonl files found to upgrade.")
        return

    for filepath in files_to_process:
        upgrade_no_reasoning_file(filepath)
        
    print("\n--- Upgrade process complete. ---")


if __name__ == "__main__":
    main()