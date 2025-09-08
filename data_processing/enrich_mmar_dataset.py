# data_processing/enrich_mmar_dataset.py

import os
import json
import argparse
import collections

def main():
    """
    This script "rescues" the MMAR dataset by enriching the standardized JSONL file
    with the detailed metadata (category, modality, etc.) from the MMAR-meta.json file.
    It uses a robust matching strategy to ensure the highest possible accuracy.
    """
    parser = argparse.ArgumentParser(description="Enrich the MMAR standardized dataset with metadata.")
    parser.add_argument('--meta-file', type=str, default='data/mmar/MMAR-meta.json', help="Path to the MMAR-meta.json file.")
    parser.add_argument('--standardized-file', type=str, default='data/mmar/mmar_test_standardized.jsonl', help="Path to the standardized JSONL file to be enriched.")
    args = parser.parse_args()

    print("--- Starting MMAR Data Enrichment Process ---")

    # --- Step 1: Build a Smart Lookup Table from the Metadata ---
    # We're going to create a dictionary where the key is the question text.
    # This is more complex than a simple dict, because some questions might be duplicated.
    print(f"Loading metadata from '{args.meta_file}'...")
    try:
        with open(args.meta_file, 'r') as f:
            meta_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Metadata file not found at '{args.meta_file}'. Exiting.")
        return
    except json.JSONDecodeError:
        print(f"FATAL: Could not parse the JSON in '{args.meta_file}'. Please check the file for errors.")
        return

    metadata_lookup = {}
    print("Building a lookup table with duplicate handling...")
    for item in meta_data:
        question = item.get('question')
        if not question:
            continue # Skip entries that don't have a question.

        if question not in metadata_lookup:
            # If this is the first time we've seen this question, just add the item.
            metadata_lookup[question] = item
        else:
            # If we've seen this question before, we have a duplicate!
            # We need to convert the entry into a list to store all possible matches.
            if not isinstance(metadata_lookup[question], list):
                # This is the second time we've seen it, so we create a list
                # containing the one we already stored.
                metadata_lookup[question] = [metadata_lookup[question]]
            
            # Add the new duplicate item to the list.
            metadata_lookup[question].append(item)
    print(f"Lookup table built. Found {len(metadata_lookup)} unique questions.")

    # --- Step 2: Load the Standardized Data to be Enriched ---
    print(f"Loading standardized data from '{args.standardized_file}'...")
    try:
        with open(args.standardized_file, 'r') as f:
            standardized_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"FATAL: Standardized file not found at '{args.standardized_file}'. Exiting.")
        return

    # --- Step 3: Iterate and Enrich Each Data Point ---
    print("Enriching data... This may take a moment.")
    enriched_data = []
    unmatched_ids = []

    for sample in standardized_data:
        question = sample.get('question')
        metadata_entry = metadata_lookup.get(question)

        found_meta = None
        if metadata_entry:
            if isinstance(metadata_entry, list):
                # This is our duplicate case. We need to use the choices as a tie-breaker.
                # We sort the choices alphabetically to make the comparison robust to order differences.
                sample_choices_sorted = sorted(sample.get('choices', []))
                
                for potential_meta in metadata_entry:
                    meta_choices_sorted = sorted(potential_meta.get('choices', []))
                    if sample_choices_sorted == meta_choices_sorted:
                        # We found the exact match!
                        found_meta = potential_meta
                        break
            else:
                # This was a unique question, so the entry is the correct one.
                found_meta = metadata_entry
        
        if found_meta:
            # If we found a match, create a new, enriched sample.
            new_sample = sample.copy()
            # We add only the keys that are not already in the sample.
            for key, value in found_meta.items():
                if key not in new_sample:
                    new_sample[key] = value
            enriched_data.append(new_sample)
        else:
            # If we couldn't find a match, we keep the original sample and log the failure.
            unmatched_ids.append(sample.get('id', 'UNKNOWN_ID'))
            enriched_data.append(sample)

    # --- Step 4: Safely Back Up the Old File and Save the New One ---
    backup_path = args.standardized_file + ".backup"
    print(f"\nEnrichment complete. Backing up original file to '{backup_path}'...")
    try:
        if os.path.exists(backup_path):
            print(f"  - WARNING: Backup file '{backup_path}' already exists. It will be overwritten.")
        os.rename(args.standardized_file, backup_path)
    except Exception as e:
        print(f"FATAL: Could not create backup file. Aborting to prevent data loss. Error: {e}")
        return

    print(f"Saving new, enriched data to '{args.standardized_file}'...")
    with open(args.standardized_file, 'w') as f:
        for entry in enriched_data:
            # Use ensure_ascii=False to keep non-English characters readable.
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # --- Step 5: Final Report ---
    print("\n--- Enrichment Summary ---")
    print(f"Total samples processed: {len(standardized_data)}")
    print(f"Successfully enriched: {len(standardized_data) - len(unmatched_ids)}")
    print(f"Unmatched samples: {len(unmatched_ids)}")
    if unmatched_ids:
        print("  - The following sample IDs could not be matched:")
        for uid in unmatched_ids:
            print(f"    - {uid}")
    print("--------------------------")
    print("Process complete.")


if __name__ == "__main__":
    main()