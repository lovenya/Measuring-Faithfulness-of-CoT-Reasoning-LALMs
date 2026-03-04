#!/usr/bin/env python3
import json
import sys
import os
import glob
from collections import defaultdict

def clean_file(filepath):
    """
    Reads a random filler JSONL file.
    For each (id, chain_id), keeps only the last 21 inferences.
    Overwrites the file with the cleaned version.
    """
    print(f"Cleaning {filepath}...")
    
    # Group lines by (id, chain_id)
    grouped_lines = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            sample_id = data.get('id')
            chain_id = data.get('chain_id')
            if sample_id is not None and chain_id is not None:
                grouped_lines[(sample_id, chain_id)].append(line)
    
    # Write back keeping only the last 21 for each group
    # We maintain the original order of the groups (order of first appearance)
    cleaned_lines = []
    seen_groups = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            key = (data.get('id'), data.get('chain_id'))
            
            if key not in seen_groups:
                seen_groups.add(key)
                
                # Keep only the last 21 (or fewer if there are less than 21)
                lines_to_keep = grouped_lines[key][-21:]
                cleaned_lines.extend(lines_to_keep)
                
                if len(grouped_lines[key]) > 21:
                    print(f"  Fixed ID: {key[0]}, Chain: {key[1]} (removed {len(grouped_lines[key]) - 21} duplicates)")

    # Atomic write-back
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    os.rename(temp_path, filepath)
    print(f"  Done. Wrote {len(cleaned_lines)} lines.\n")

if __name__ == '__main__':
    # Find all filler part files
    files = glob.glob('results/*/random_partial_filler_text/*.part_*.jsonl')
    for f in sorted(files):
        # Quick check if it needs cleaning
        needs_cleaning = False
        with open(f, 'r') as file:
            counts = defaultdict(int)
            for line in file:
                d = json.loads(line)
                counts[(d.get('id'), d.get('chain_id'))] += 1
                if counts[(d.get('id'), d.get('chain_id'))] > 21:
                    needs_cleaning = True
                    break
        
        if needs_cleaning:
            clean_file(f)
        else:
            print(f"OK: {f} (No duplicates)")
