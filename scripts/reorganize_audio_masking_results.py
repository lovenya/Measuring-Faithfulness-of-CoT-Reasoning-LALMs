#!/usr/bin/env python3
"""
Reorganize audio masking results from flat structure to hierarchical.

Before: results/qwen/audio_masking/audio_masking_qwen_mmar_silence_start.jsonl
After:  results/qwen/audio_masking/silence/start/audio_masking_qwen_mmar_silence_start.jsonl

Also cleans up:
- Old combined files (no mask_type/mode in name)
- Random mode files (--delete-random flag)
"""

import argparse
import os
import shutil
from pathlib import Path
import re


def reorganize(model: str, results_dir: str = "results", delete_random: bool = False, dry_run: bool = False):
    audio_dir = Path(results_dir) / model / "audio_masking"
    
    if not audio_dir.exists():
        print(f"ERROR: {audio_dir} not found")
        return
    
    mask_types = ["silence", "noise"]
    mask_modes = ["start", "end", "random", "scattered"]
    
    # Find all JSONL files in the flat directory
    moved = 0
    deleted = 0
    
    for f in sorted(audio_dir.glob("*.jsonl")):
        name = f.name
        
        # Detect mask_type and mask_mode from filename
        detected_type = None
        detected_mode = None
        
        for mt in mask_types:
            for mm in mask_modes:
                if f"_{mt}_{mm}" in name:
                    detected_type = mt
                    detected_mode = mm
                    break
            if detected_type:
                break
        
        if not detected_type:
            # This is an old combined file (no mask_type in name)
            if dry_run:
                print(f"  [DRY RUN] DELETE (old combined): {name}")
            else:
                f.unlink()
                print(f"  DELETED (old combined): {name}")
            deleted += 1
            continue
        
        # Handle random mode files
        if detected_mode == "random" and delete_random:
            if dry_run:
                print(f"  [DRY RUN] DELETE (random): {name}")
            else:
                f.unlink()
                print(f"  DELETED (random): {name}")
            deleted += 1
            continue
        
        # Move to hierarchical structure
        target_dir = audio_dir / detected_type / detected_mode
        target_path = target_dir / name
        
        if dry_run:
            print(f"  [DRY RUN] MOVE: {name} → {detected_type}/{detected_mode}/")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(target_path))
            print(f"  MOVED: {name} → {detected_type}/{detected_mode}/")
        moved += 1
    
    print(f"\nSummary: {moved} moved, {deleted} deleted")


def main():
    parser = argparse.ArgumentParser(description="Reorganize audio masking results into hierarchical folders.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'qwen')")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--delete-random", action="store_true", help="Delete random mode files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Audio Masking Results Reorganizer")
    print("=" * 60)
    
    reorganize(args.model, args.results_dir, args.delete_random, args.dry_run)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
