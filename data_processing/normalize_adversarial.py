# data_processing/normalize_adversarial.py

"""
Generates standardized JSONL files for the adversarial audio experiment.

For each of the 8 adversarial subfolders (4 tracks × {concat, overlay}), this
script produces 2 JSONLs (correct + wrong), for a total of 16 output files.

Each output JSONL mirrors the existing sakura standardized format (question,
choices, answer_key, etc.) but replaces the audio_path with the adversarial
version (correct.wav or wrong.wav).

The script reads the augment_inject_manifest.jsonl from each subfolder to get
the mapping from sample IDs to adversarial audio paths, and matches them against
the corresponding sakura standardized JSONL.

Note: language_overlay has duplicated entries (1000 unique IDs × 2 = 2000 lines).
This script automatically deduplicates by ID.
"""

import os
import json
import argparse


TRACKS = ['animal', 'emotion', 'gender', 'language']
AUG_MODES = ['concat', 'overlay']
VARIANTS = ['correct', 'wrong']

# Maps track name to the corresponding sakura standardized JSONL
SAKURA_JSONL_MAP = {
    'animal': 'data/sakura/animal/sakura_animal_test_standardized.jsonl',
    'emotion': 'data/sakura/emotion/sakura_emotion_test_standardized.jsonl',
    'gender': 'data/sakura/gender/sakura_gender_test_standardized.jsonl',
    'language': 'data/sakura/language/sakura_language_test_standardized.jsonl',
}


def load_sakura_data(track: str, project_root: str) -> dict:
    """Load sakura standardized JSONL and return a dict keyed by sample ID."""
    jsonl_path = os.path.join(project_root, SAKURA_JSONL_MAP[track])
    data_by_id = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            data_by_id[sample['id']] = sample
    print(f"  Loaded {len(data_by_id)} samples from {jsonl_path}")
    return data_by_id


def load_manifest(subfolder_path: str) -> dict:
    """
    Load the augment_inject_manifest.jsonl and return a dict keyed by sample ID.
    Automatically deduplicates by keeping only the first occurrence of each ID.
    """
    manifest_path = os.path.join(subfolder_path, 'augment_inject_manifest.jsonl')
    manifest_by_id = {}
    total_lines = 0
    with open(manifest_path, 'r') as f:
        for line in f:
            total_lines += 1
            entry = json.loads(line)
            sample_id = entry['id']
            if sample_id not in manifest_by_id:
                manifest_by_id[sample_id] = entry
    
    if total_lines != len(manifest_by_id):
        print(f"  ⚠ Deduplicated: {total_lines} lines → {len(manifest_by_id)} unique IDs")
    else:
        print(f"  Loaded {len(manifest_by_id)} entries from manifest")
    
    return manifest_by_id


def generate_adversarial_jsonl(
    track: str, 
    aug_mode: str, 
    variant: str, 
    sakura_data: dict, 
    manifest: dict,
    project_root: str
) -> str:
    """
    Generate a standardized JSONL for one (track, aug_mode, variant) combination.
    Returns the output path.
    """
    subfolder = f"{track}_{aug_mode}"
    output_filename = f"adversarial_{track}_{aug_mode}_{variant}.jsonl"
    output_path = os.path.join(
        project_root, 'data', 'adversarial_aug_data', subfolder, output_filename
    )
    
    matched = 0
    missing = 0
    
    with open(output_path, 'w') as f:
        for sample_id, manifest_entry in manifest.items():
            if sample_id not in sakura_data:
                missing += 1
                continue
            
            sakura_sample = sakura_data[sample_id]
            
            # Build the adversarial audio path from the manifest
            audio_path = manifest_entry[f'out_{variant}']
            
            # Fix path inconsistency: some manifests use 'data/adversarial_data' 
            # while the files are actually in 'data/adversarial_aug_data'
            if 'data/adversarial_data/' in audio_path:
                audio_path = audio_path.replace('data/adversarial_data/', 'data/adversarial_aug_data/')
            
            # Create the output sample: same question/choices/answer, different audio
            output_sample = {
                'id': sample_id,
                'audio_path': audio_path,
                'question': sakura_sample['question'],
                'choices': sakura_sample['choices'],
                'answer': sakura_sample['answer'],
                'answer_key': sakura_sample['answer_key'],
                'hop_type': sakura_sample.get('hop_type', ''),
                'track': sakura_sample.get('track', track),
                'modality': sakura_sample.get('modality', 'audio'),
                'language': sakura_sample.get('language', 'en'),
                'source': sakura_sample.get('source', 'sakura'),
                'adversarial_aug': aug_mode,
                'adversarial_variant': variant,
            }
            
            f.write(json.dumps(output_sample, ensure_ascii=False) + '\n')
            matched += 1
    
    if missing > 0:
        print(f"  ⚠ {missing} manifest IDs not found in sakura data!")
    
    print(f"  ✓ Wrote {matched} samples → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate standardized JSONLs for adversarial audio experiments."
    )
    parser.add_argument(
        '--project-root', type=str, default='.',
        help="Path to the project root directory."
    )
    parser.add_argument(
        '--tracks', type=str, nargs='+', default=TRACKS,
        choices=TRACKS,
        help="Which tracks to process (default: all)."
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Adversarial Audio Data Normalization")
    print("=" * 60)
    
    total_files = 0
    
    for track in args.tracks:
        print(f"\n--- Track: {track.upper()} ---")
        
        # Load the sakura data for this track once
        sakura_data = load_sakura_data(track, args.project_root)
        
        for aug_mode in AUG_MODES:
            subfolder = f"{track}_{aug_mode}"
            subfolder_path = os.path.join(
                args.project_root, 'data', 'adversarial_aug_data', subfolder
            )
            
            if not os.path.exists(subfolder_path):
                print(f"  ⚠ Subfolder not found: {subfolder_path}, skipping.")
                continue
            
            print(f"\n  Subfolder: {subfolder}")
            manifest = load_manifest(subfolder_path)
            
            for variant in VARIANTS:
                print(f"  Generating: {variant}")
                generate_adversarial_jsonl(
                    track, aug_mode, variant, sakura_data, manifest, args.project_root
                )
                total_files += 1
    
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {total_files} standardized JSONL files.")
    print("=" * 60)


if __name__ == '__main__':
    main()
