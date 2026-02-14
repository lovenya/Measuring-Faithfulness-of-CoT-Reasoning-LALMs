#!/usr/bin/env python3
"""
Utility script to split combined audio masking JSONL files into separate files
by mask_type and mask_mode.

Before: audio_masking_qwen_mmar.jsonl (contains all mask_type × mask_mode combinations)
After:  audio_masking_qwen_mmar_silence_start.jsonl
        audio_masking_qwen_mmar_silence_end.jsonl
        audio_masking_qwen_mmar_silence_random.jsonl  (or scattered)
        audio_masking_qwen_mmar_noise_start.jsonl
        audio_masking_qwen_mmar_noise_end.jsonl
        audio_masking_qwen_mmar_noise_random.jsonl  (or scattered)

Usage:
    python split_audio_masking_results.py --input results/qwen/audio_masking/audio_masking_qwen_mmar.jsonl
    
    # Process all audio masking files for a model:
    python split_audio_masking_results.py --model qwen --all
    
    # Delete random mode entries after splitting:
    python split_audio_masking_results.py --model qwen --all --delete-random
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


def split_jsonl_file(input_path: str, delete_random: bool = False, dry_run: bool = False):
    """
    Split a combined audio masking JSONL into separate files by mask_type and mask_mode.
    
    Args:
        input_path: Path to the combined JSONL file
        delete_random: If True, skip random mode entries (don't write them)
        dry_run: If True, only print what would be done without writing
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"  ERROR: File not found: {input_path}")
        return
    
    # Parse the base filename to extract model and dataset
    # Expected format: audio_masking_<model>_<dataset>.jsonl
    stem = input_path.stem  # audio_masking_qwen_mmar
    parts = stem.split('_')
    if len(parts) < 4 or parts[0] != 'audio' or parts[1] != 'masking':
        print(f"  ERROR: Unexpected filename format: {input_path.name}")
        return
    
    model = parts[2]
    dataset = '_'.join(parts[3:])  # Handle datasets with underscores like sakura-emotion
    
    output_dir = input_path.parent
    
    # Group entries by (mask_type, mask_mode)
    grouped_entries = defaultdict(list)
    total_entries = 0
    skipped_random = 0
    
    with open(input_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                mask_type = entry.get('mask_type', 'unknown')
                mask_mode = entry.get('mask_mode', 'unknown')
                
                # Skip random mode if requested
                if delete_random and mask_mode == 'random':
                    skipped_random += 1
                    continue
                
                grouped_entries[(mask_type, mask_mode)].append(entry)
                total_entries += 1
            except json.JSONDecodeError:
                print(f"  WARNING: Skipping malformed line")
                continue
    
    print(f"\n  Input: {input_path.name}")
    print(f"  Total entries: {total_entries}")
    if delete_random:
        print(f"  Skipped random entries: {skipped_random}")
    print(f"  Groups found: {len(grouped_entries)}")
    
    # Write separate files for each group
    for (mask_type, mask_mode), entries in grouped_entries.items():
        output_filename = f"audio_masking_{model}_{dataset}_{mask_type}_{mask_mode}.jsonl"
        output_path = output_dir / output_filename
        
        if dry_run:
            print(f"    [DRY RUN] Would write {len(entries)} entries to {output_filename}")
        else:
            with open(output_path, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')
            print(f"    Wrote {len(entries)} entries to {output_filename}")
    
    return grouped_entries


def find_combined_files(model: str, results_dir: str = "results") -> list:
    """
    Find all combined audio masking JSONL files for a model.
    These are files that DON'T have mask_type/mask_mode in their name.
    """
    results_path = Path(results_dir) / model / "audio_masking"
    if not results_path.exists():
        print(f"  ERROR: Directory not found: {results_path}")
        return []
    
    combined_files = []
    for f in results_path.glob("audio_masking_*.jsonl"):
        # Skip files that already have mask_type in name (already split)
        name = f.stem
        # Combined files: audio_masking_qwen_mmar.jsonl
        # Split files: audio_masking_qwen_mmar_silence_start.jsonl
        parts = name.split('_')
        # If the file has more than 4 parts and one of them is a mask_type, it's already split
        if any(part in ['silence', 'noise'] for part in parts[4:]):
            continue
        combined_files.append(f)
    
    return combined_files


def main():
    parser = argparse.ArgumentParser(
        description="Split combined audio masking JSONL files into separate files by mask_type and mask_mode.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', type=str, 
        help="Path to a specific combined JSONL file to split")
    parser.add_argument('--model', type=str, 
        help="Model name (e.g., 'qwen', 'salmonn_7b') - used with --all")
    parser.add_argument('--all', action='store_true',
        help="Process all combined audio masking files for the specified model")
    parser.add_argument('--delete-random', action='store_true',
        help="Skip/delete random mode entries (don't write them to split files)")
    parser.add_argument('--dry-run', action='store_true',
        help="Only print what would be done without writing files")
    parser.add_argument('--results-dir', type=str, default='results',
        help="Base results directory (default: results)")
    
    args = parser.parse_args()
    
    if not args.input and not (args.model and args.all):
        parser.error("Must specify either --input or both --model and --all")
    
    print("=" * 60)
    print("Audio Masking JSONL Splitter")
    print("=" * 60)
    
    if args.input:
        # Process single file
        split_jsonl_file(args.input, args.delete_random, args.dry_run)
    
    elif args.model and args.all:
        # Process all combined files for model
        combined_files = find_combined_files(args.model, args.results_dir)
        
        if not combined_files:
            print(f"  No combined audio masking files found for model '{args.model}'")
            return
        
        print(f"\nFound {len(combined_files)} combined file(s) for model '{args.model}':")
        for f in combined_files:
            print(f"  - {f.name}")
        
        for f in combined_files:
            split_jsonl_file(f, args.delete_random, args.dry_run)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
