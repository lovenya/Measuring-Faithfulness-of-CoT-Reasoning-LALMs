#!/usr/bin/env python3
"""
Mask the JASCO dataset directly from the source v0 dataset.
This script:
1. Loads ground truth (v0.csv) and prompts (example.csv).
2. Loads speech segment timestamps (speech_segments.jsonl).
3. Applies scattered silence masking to ONLY the Speech region (10% to 100%)
   AND ONLY the Audio region (10% to 100%). Total length remains unchanged.
4. Uses a reproducible random seed strategy.
5. Outputs a standardized JSONL for the JASCO masking experiments.

Usage:
    python data_processing/mask_jasco.py --workers 8
"""

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import soundfile as sf

# Import global seed from config so it's a single source of truth
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
    GLOBAL_SEED = config.JASCO_GLOBAL_SEED
except ImportError:
    print("Warning: Could not import config. Using fallback seed 42.")
    GLOBAL_SEED = 42

def parse_timestamp(ts_str: str) -> tuple:
    """Parse a timestamp string like '8.1s --> 9.9s' into (start, end) in seconds."""
    parts = ts_str.split('-->')
    start = float(parts[0].strip().rstrip('s'))
    end = float(parts[1].strip().rstrip('s'))
    return start, end

def map_virtual_to_physical(start_v, end_v, valid_regions):
    """
    Given a virtual segment [start_v, end_v] within a sequence of contiguous
    virtual blocks derived from valid_regions, return the corresponding
    physical segments.
    """
    segments = []
    current_v = 0
    for s, e in valid_regions:
        length = e - s
        region_v_start = current_v
        region_v_end = current_v + length
        
        # Check overlap
        overlap_start = max(start_v, region_v_start)
        overlap_end = min(end_v, region_v_end)
        
        if overlap_start < overlap_end:
            # Map back to physical
            phys_start = s + (overlap_start - region_v_start)
            phys_end = s + (overlap_end - region_v_start)
            segments.append((phys_start, phys_end))
            
        current_v += length
        
    return segments

def get_scattered_mask_ranges(valid_regions: list, percent: int, seed_str: str, 
                               min_segment_ms: int = 50, max_segment_ms: int = 500, 
                               sample_rate: int = 16000) -> list:
    """
    Generates scattered silence mask physical segments specifically restricted
    inside the `valid_regions`.
    """
    total_valid_samples = sum(e - s for s, e in valid_regions)
    total_mask_samples = int(total_valid_samples * percent / 100)
    
    if total_mask_samples == 0:
        return []
        
    min_segment_samples = int(min_segment_ms * sample_rate / 1000)
    max_segment_samples = int(max_segment_ms * sample_rate / 1000)
    
    max_segment_samples = min(max_segment_samples, total_mask_samples)
    if max_segment_samples == 0:
        return []
    min_segment_samples = min(min_segment_samples, max_segment_samples)
    
    rng = random.Random(seed_str)
    
    remaining_mask = total_mask_samples
    masked_virtual_regions = []
    
    max_attempts = 1000
    attempts = 0
    
    while remaining_mask >= min_segment_samples and attempts < max_attempts:
        attempts += 1
        seg_len = min(rng.randint(min_segment_samples, max_segment_samples), remaining_mask)
        
        max_start = total_valid_samples - seg_len
        if max_start <= 0:
            break
            
        start_v = rng.randint(0, max_start)
        end_v = start_v + seg_len
        
        overlaps = False
        for existing_start, existing_end in masked_virtual_regions:
            if not (end_v <= existing_start or start_v >= existing_end):
                overlaps = True
                break
                
        if not overlaps:
            masked_virtual_regions.append((start_v, end_v))
            remaining_mask -= seg_len
            
    physical_segments = []
    for start_v, end_v in masked_virtual_regions:
        curr_phys_segments = map_virtual_to_physical(start_v, end_v, valid_regions)
        physical_segments.extend(curr_phys_segments)
        
    physical_segments.sort(key=lambda x: x[0])
    return physical_segments

def mask_audio_with_silence(audio: np.ndarray, physical_segments: list) -> np.ndarray:
    masked_audio = audio.copy()
    for start, end in physical_segments:
        if masked_audio.ndim == 2:
            masked_audio[start:end, :] = 0.0
        else:
            masked_audio[start:end] = 0.0
    return masked_audio

def process_single_audio(aid, gt, prompts, speech_start_sec, speech_end_sec, audio_path, output_dir_processed):
    """
    Process a single audio file and generate all 21 variants.
    """
    if not audio_path.exists():
        return None
        
    audio, sr = sf.read(audio_path, dtype='float32')
    total_samples = len(audio) if audio.ndim == 1 else audio.shape[0]
    
    s_start = int(speech_start_sec * sr)
    s_end = int(speech_end_sec * sr)
    
    # Clamp to valid range inside the audio
    s_start = max(0, min(s_start, total_samples))
    s_end = max(0, min(s_end, total_samples))
    
    # Define Valid Regions for Sub-modalities
    speech_region = [(s_start, s_end)]
    
    audio_region = []
    if s_start > 0:
        audio_region.append((0, s_start))
    if s_end < total_samples:
        audio_region.append((s_end, total_samples))
        
    levels = list(range(10, 101, 10))
    paths_dict = {}
    
    # 0. Baseline (0% masked)
    baseline_dir = output_dir_processed / 'full_audio'
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = baseline_dir / f'{aid}.wav'
    if not baseline_path.exists():
        sf.write(str(baseline_path), audio, sr)
    paths_dict['baseline'] = str(baseline_path)
    
    # 1. Mask Speech Segment (10% to 100%)
    for pct in levels:
        subdir = output_dir_processed / 'speech_masked' / str(pct)
        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / f'{aid}.wav'
        
        if not out_path.exists():
            seed_str = f"{GLOBAL_SEED}_{aid}_speech_{pct}"
            segments_to_mask = get_scattered_mask_ranges(speech_region, pct, seed_str, sample_rate=sr)
            masked_audio = mask_audio_with_silence(audio, segments_to_mask)
            sf.write(str(out_path), masked_audio, sr)
            
        paths_dict[f'speech_{pct}'] = str(out_path)
        
    # 2. Mask Audio Segment (10% to 100%)
    for pct in levels:
        subdir = output_dir_processed / 'audio_masked' / str(pct)
        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / f'{aid}.wav'
        
        if not out_path.exists():
            seed_str = f"{GLOBAL_SEED}_{aid}_audio_{pct}"
            segments_to_mask = get_scattered_mask_ranges(audio_region, pct, seed_str, sample_rate=sr)
            masked_audio = mask_audio_with_silence(audio, segments_to_mask)
            sf.write(str(out_path), masked_audio, sr)
            
        paths_dict[f'audio_{pct}'] = str(out_path)
        
    # Compile the final entry
    entry = {
        'id': aid,
        'audio_paths': paths_dict,
        'prompts': prompts,
        'audio_sound': gt['audio_sound'],
        'spoken_text': gt['spoken_text'],
        'target_keywords': gt['target_keywords'],
        'correct_answer': gt['correct_answer'],
        'audio_only_answer': gt['audio_only_answer'],
        'speech_only_answer': gt['speech_only_answer'],
    }
    return entry


def main():
    parser = argparse.ArgumentParser(description='Process JASCO tailored data')
    parser.add_argument('--jasco-dir', type=str, default='data/jasco',
                        help='Path to JASCO root directory')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of multiprocessing workers')
    args = parser.parse_args()
    
    jasco_dir = Path(args.jasco_dir)
    v0_csv = jasco_dir / 'dataset' / 'v0' / 'v0.csv'
    example_csv = jasco_dir / 'evaluation' / 'example.csv'
    segments_jsonl = jasco_dir / 'speech_segments.jsonl'
    
    output_dir_processed = jasco_dir / 'processed'
    output_jsonl = jasco_dir / 'jasco_masked_standardized.jsonl'
    
    # 1. Load Ground Truth
    print(f"Loading ground truth from {v0_csv}...")
    ground_truth = {}
    with open(v0_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row['audio_id'].strip()
            keywords_raw = row['Correct Answer Keyword'].strip()
            keywords = [k.strip() for k in keywords_raw.split(',')]
            ground_truth[aid] = {
                'audio_sound': row['Audio'].strip(),
                'spoken_text': row['Speech'].strip(),
                'audio_only_answer': row['Audio-Only Answer'].strip(),
                'speech_only_answer': row['Speech-Only Answer'].strip(),
                'correct_answer': row['Correct Answer'].strip(),
                'target_keywords': keywords,
            }
            
    # 2. Load Prompts
    print(f"Loading prompts from {example_csv}...")
    prompts_by_id = {}
    with open(example_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row['id'].strip()
            prompt = row['prompt'].strip()
            if aid not in prompts_by_id:
                prompts_by_id[aid] = []
            if prompt not in prompts_by_id[aid]:
                prompts_by_id[aid].append(prompt)
                
    # 3. Load Speech Segments
    print(f"Loading speech segments from {segments_jsonl}...")
    segments = {}
    with open(segments_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_id = Path(entry['audio_path']).stem
            speech_start, speech_end = parse_timestamp(entry['speech_segment'])
            segments[audio_id] = (speech_start, speech_end)
            
    # 4. Process the data
    print(f"\nProcessing audio files using {args.workers} workers...")
    print(f"Using JASCO_GLOBAL_SEED: {GLOBAL_SEED}")
    output_dir_processed.mkdir(parents=True, exist_ok=True)
    
    audio_dir = jasco_dir / 'dataset' / 'v0'
    entries = []
    
    aids_to_process = sorted(ground_truth.keys(), key=int)
    
    if args.workers > 1:
        max_workers = min(args.workers, multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for aid in aids_to_process:
                audio_path = audio_dir / f'{aid}.wav'
                if aid not in segments:
                    continue
                s_start, s_end = segments[aid]
                
                futures.append(
                    executor.submit(
                        process_single_audio, aid, ground_truth[aid], prompts_by_id.get(aid, []),
                        s_start, s_end, audio_path, output_dir_processed
                    )
                )
                
            for i, future in enumerate(as_completed(futures)):
                res = future.result()
                if res:
                    entries.append(res)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(futures)}")
    else:
        for i, aid in enumerate(aids_to_process):
            audio_path = audio_dir / f'{aid}.wav'
            if aid not in segments:
                continue
            s_start, s_end = segments[aid]
            res = process_single_audio(
                aid, ground_truth[aid], prompts_by_id.get(aid, []),
                s_start, s_end, audio_path, output_dir_processed
            )
            if res:
                entries.append(res)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(aids_to_process)}")
                
    # Sort entries nicely
    entries.sort(key=lambda x: int(x['id']))
    
    with open(output_jsonl, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\n{'='*60}")
    print(f"JASCO normalization complete!")
    print(f"  Entries processed: {len(entries)}")
    print(f"  Output JSONL:      {output_jsonl}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
