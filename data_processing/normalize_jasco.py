#!/usr/bin/env python3
"""
Normalize the JASCO (What Are They Doing) dataset.

This script:
1. Reads v0.csv (ground truth), example.csv (prompts), and speech_segments.jsonl (timestamps)
2. Trims each audio clip into 3 variants:
   - full/     : original audio (copied)
   - audio_only/: speech segment removed, environmental audio only
   - speech_only/: only the speech segment retained
3. Outputs a standardized JSONL for use with the jasco_masking experiment

Usage:
    python data_processing/normalize_jasco.py [--jasco-dir data/jasco]
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import soundfile as sf
import numpy as np


def parse_timestamp(ts_str: str) -> tuple:
    """Parse a timestamp string like '8.1s --> 9.9s' into (start, end) in seconds."""
    parts = ts_str.split('-->')
    start = float(parts[0].strip().rstrip('s'))
    end = float(parts[1].strip().rstrip('s'))
    return start, end


def trim_audio(audio_path: str, speech_start: float, speech_end: float, output_dir: Path, audio_id: str):
    """
    Create 3 audio variants from the original:
    - full: original audio (copy)
    - audio_only: speech segment removed (before + after spliced)
    - speech_only: only the speech segment
    """
    audio, sr = sf.read(audio_path, dtype='float32')
    
    start_sample = int(speech_start * sr)
    end_sample = int(speech_end * sr)
    
    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    # 1. Full: copy original
    full_dir = output_dir / 'full'
    full_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / f'{audio_id}.wav'
    if not full_path.exists():
        shutil.copy2(audio_path, full_path)
    
    # 2. Audio-only: remove speech segment (splice before + after)
    audio_only_dir = output_dir / 'audio_only'
    audio_only_dir.mkdir(parents=True, exist_ok=True)
    audio_only_path = audio_only_dir / f'{audio_id}.wav'
    if not audio_only_path.exists():
        before_speech = audio[:start_sample]
        after_speech = audio[end_sample:]
        audio_only = np.concatenate([before_speech, after_speech])
        if len(audio_only) > 0:
            sf.write(str(audio_only_path), audio_only, sr)
        else:
            # Edge case: entire audio is speech — save silence
            sf.write(str(audio_only_path), np.zeros(sr, dtype='float32'), sr)
    
    # 3. Speech-only: keep only the speech segment
    speech_only_dir = output_dir / 'speech_only'
    speech_only_dir.mkdir(parents=True, exist_ok=True)
    speech_only_path = speech_only_dir / f'{audio_id}.wav'
    if not speech_only_path.exists():
        speech_only = audio[start_sample:end_sample]
        if len(speech_only) > 0:
            sf.write(str(speech_only_path), speech_only, sr)
        else:
            # Edge case: no speech found — save silence
            sf.write(str(speech_only_path), np.zeros(sr, dtype='float32'), sr)
    
    return str(full_path), str(audio_only_path), str(speech_only_path)


def main():
    parser = argparse.ArgumentParser(description='Normalize JASCO dataset')
    parser.add_argument('--jasco-dir', type=str, default='data/jasco',
                        help='Path to the JASCO dataset directory')
    args = parser.parse_args()
    
    jasco_dir = Path(args.jasco_dir)
    v0_csv = jasco_dir / 'dataset' / 'v0' / 'v0.csv'
    example_csv = jasco_dir / 'evaluation' / 'example.csv'
    segments_jsonl = jasco_dir / 'speech_segments.jsonl'
    output_dir = jasco_dir / 'processed'
    output_jsonl = jasco_dir / 'jasco_standardized.jsonl'
    
    # --- 1. Load ground truth from v0.csv ---
    print(f"Loading ground truth from {v0_csv}...")
    ground_truth = {}
    with open(v0_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row['audio_id'].strip()
            keywords_raw = row['Correct Answer Keyword'].strip()
            # Split multi-keyword entries like "sport,referee"
            keywords = [k.strip() for k in keywords_raw.split(',')]
            ground_truth[aid] = {
                'audio_sound': row['Audio'].strip(),
                'spoken_text': row['Speech'].strip(),
                'audio_only_answer': row['Audio-Only Answer'].strip(),
                'speech_only_answer': row['Speech-Only Answer'].strip(),
                'correct_answer': row['Correct Answer'].strip(),
                'target_keywords': keywords,
            }
    print(f"  Loaded {len(ground_truth)} ground truth entries")
    
    # --- 2. Load prompts from example.csv ---
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
    print(f"  Loaded prompts for {len(prompts_by_id)} audio IDs")
    for aid in list(prompts_by_id.keys())[:3]:
        print(f"    ID {aid}: {len(prompts_by_id[aid])} unique prompts")
    
    # --- 3. Load speech segments ---
    print(f"Loading speech segments from {segments_jsonl}...")
    segments = {}
    with open(segments_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Extract audio ID from path like "What_Are_They_Doing/dataset/v0/43.wav"
            audio_id = Path(entry['audio_path']).stem
            speech_start, speech_end = parse_timestamp(entry['speech_segment'])
            segments[audio_id] = (speech_start, speech_end)
    print(f"  Loaded {len(segments)} speech segment timestamps")
    
    # --- 4. Process audio and create standardized JSONL ---
    print(f"\nProcessing audio files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    entries = []
    audio_dir = jasco_dir / 'dataset' / 'v0'
    
    # Process in sorted order for reproducibility
    for aid in sorted(ground_truth.keys(), key=int):
        audio_path = audio_dir / f'{aid}.wav'
        
        if not audio_path.exists():
            print(f"  WARNING: Audio file not found: {audio_path}")
            continue
        
        if aid not in segments:
            print(f"  WARNING: No speech segment for audio ID {aid}")
            continue
        
        speech_start, speech_end = segments[aid]
        
        # Trim audio into 3 variants
        full_path, audio_only_path, speech_only_path = trim_audio(
            str(audio_path), speech_start, speech_end, output_dir, aid
        )
        
        gt = ground_truth[aid]
        prompts = prompts_by_id.get(aid, [])
        
        entry = {
            'id': aid,
            'audio_path': full_path,               # original full audio
            'audio_only_path': audio_only_path,     # environmental audio only
            'speech_only_path': speech_only_path,   # speech only
            'prompts': prompts,                     # all 8 prompt variants
            'audio_sound': gt['audio_sound'],
            'spoken_text': gt['spoken_text'],
            'target_keywords': gt['target_keywords'],
            'correct_answer': gt['correct_answer'],
            'audio_only_answer': gt['audio_only_answer'],
            'speech_only_answer': gt['speech_only_answer'],
        }
        entries.append(entry)
        
        if int(aid) % 20 == 0:
            print(f"  Processed {aid}/80")
    
    # Write standardized JSONL
    with open(output_jsonl, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n{'='*60}")
    print(f"JASCO normalization complete!")
    print(f"  Entries:     {len(entries)}")
    print(f"  Output JSONL: {output_jsonl}")
    print(f"  Audio dirs:   {output_dir}/{{full,audio_only,speech_only}}/")
    
    # Quick stats
    for variant in ['full', 'audio_only', 'speech_only']:
        variant_dir = output_dir / variant
        count = len(list(variant_dir.glob('*.wav')))
        print(f"  {variant}/: {count} files")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
