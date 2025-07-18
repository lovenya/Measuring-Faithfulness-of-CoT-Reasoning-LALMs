#!/usr/bin/env python3
"""
Script to fix SAKURA dataset nomenclature:
1. Update JSONL file paths to match existing sequential audio filenames
2. Remove audio_path_status field
3. Handle 2 JSONL entries per audio file (single + multi hop questions)
"""

import os
import json
import shutil
from pathlib import Path

# Configuration
SAKURA_DATASETS = {
    "animal": {
        "jsonl_path": "./data/sakura/animal/sakura_animal_test_standardized.jsonl",
        "audio_dir": "./data/sakura/animal/audio",
        "prefix": "sakura_animal"
    },
    "emotion": {
        "jsonl_path": "./data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
        "audio_dir": "./data/sakura/emotion/audio",
        "prefix": "sakura_emotion"
    },
    "gender": {
        "jsonl_path": "./data/sakura/gender/sakura_gender_test_standardized.jsonl",
        "audio_dir": "./data/sakura/gender/audio",
        "prefix": "sakura_gender"
    },
    "language": {
        "jsonl_path": "./data/sakura/language/sakura_language_test_standardized.jsonl",
        "audio_dir": "./data/sakura/language/audio",
        "prefix": "sakura_language"
    }
}

def backup_original_jsonl(jsonl_path):
    """Create a backup of the original JSONL file"""
    backup_path = jsonl_path + ".backup"
    if os.path.exists(jsonl_path):
        shutil.copy2(jsonl_path, backup_path)
        print(f"✓ Backup created: {backup_path}")
        return True
    else:
        print(f"ERROR: Original JSONL file not found at {jsonl_path}")
        return False

def load_current_jsonl(jsonl_path):
    """Load current JSONL entries"""
    entries = []
    try:
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON at line {line_num}: {e}")
                    return None
        print(f"✓ Loaded {len(entries)} entries from JSONL")
        return entries
    except Exception as e:
        print(f"ERROR: Failed to load JSONL file: {e}")
        return None

def verify_audio_files(audio_dir, prefix, expected_count):
    """Verify that expected audio files exist"""
    if not os.path.exists(audio_dir):
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return False
    
    missing_files = []
    for i in range(expected_count):
        audio_file = f"{prefix}_{i}.wav"
        audio_path = os.path.join(audio_dir, audio_file)
        if not os.path.exists(audio_path):
            missing_files.append(audio_file)
    
    if missing_files:
        print(f"ERROR: Missing audio files in {audio_dir}:")
        for file in missing_files[:5]:  # Show first 5 missing files
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False
    
    print(f"✓ All {expected_count} audio files verified in {audio_dir}")
    return True

def update_sakura_entries(entries, dataset_name, prefix):
    """Update SAKURA JSONL entries with correct paths"""
    print(f"\n--- Updating {dataset_name.upper()} JSONL Paths ---")
    
    updated_entries = []
    for i, entry in enumerate(entries):
        # Create new entry with updated path and removed audio_path_status
        updated_entry = entry.copy()
        
        # Remove audio_path_status field if it exists
        if 'audio_path_status' in updated_entry:
            del updated_entry['audio_path_status']
        
        # Calculate audio file index (2 entries per audio file)
        audio_index = i // 2
        new_audio_path = f"data/sakura/{dataset_name}/audio/{prefix}_{audio_index}.wav"
        updated_entry['audio_path'] = new_audio_path
        
        updated_entries.append(updated_entry)
        
        # Print every 10th entry to avoid too much output
        if i % 10 == 0 or i < 5:
            old_path = entry.get('audio_path', 'N/A')
            hop_type = entry.get('hop_type', 'N/A')
            print(f"Entry {i} ({hop_type}): {old_path} -> {new_audio_path}")
    
    return updated_entries

def save_updated_jsonl(updated_entries, jsonl_path):
    """Save the updated JSONL file"""
    try:
        with open(jsonl_path, 'w') as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Updated JSONL saved to {jsonl_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save updated JSONL: {e}")
        return False

def verify_sakura_results(jsonl_path, dataset_name, prefix):
    """Verify that the path updates were successful"""
    print(f"\n--- Verification for {dataset_name.upper()} ---")
    
    # Check JSONL paths
    entries = load_current_jsonl(jsonl_path)
    if not entries:
        return False
    
    # Verify paths and structure
    for i, entry in enumerate(entries):
        expected_audio_index = i // 2
        expected_path = f"data/sakura/{dataset_name}/audio/{prefix}_{expected_audio_index}.wav"
        actual_path = entry.get('audio_path', '')
        
        if actual_path != expected_path:
            print(f"ERROR: JSONL path mismatch at entry {i}")
            print(f"Expected: {expected_path}")
            print(f"Actual: {actual_path}")
            return False
        
        # Check that audio_path_status was removed
        if 'audio_path_status' in entry:
            print(f"ERROR: audio_path_status field not removed from entry {i}")
            return False
        
        # Verify hop_type exists
        if 'hop_type' not in entry:
            print(f"ERROR: hop_type field missing from entry {i}")
            return False
    
    print(f"✓ All {len(entries)} JSONL paths updated correctly")
    print(f"✓ audio_path_status field removed from all entries")
    
    # Check that we have pairs of single/multi entries
    hop_types = [entry.get('hop_type') for entry in entries]
    expected_pattern = ['single', 'multi'] * (len(entries) // 2)
    
    if hop_types != expected_pattern:
        print(f"WARNING: Unexpected hop_type pattern")
        print(f"Expected: {expected_pattern[:10]}...")
        print(f"Actual: {hop_types[:10]}...")
    else:
        print(f"✓ Hop type pattern verified (single/multi pairs)")
    
    return True

def process_sakura_dataset(dataset_name, config):
    """Process a single SAKURA dataset"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    jsonl_path = config["jsonl_path"]
    audio_dir = config["audio_dir"]
    prefix = config["prefix"]
    
    # Step 1: Backup original JSONL
    if not backup_original_jsonl(jsonl_path):
        return False
    
    # Step 2: Load current JSONL entries
    entries = load_current_jsonl(jsonl_path)
    if not entries:
        return False
    
    # Step 3: Verify audio files exist (2 entries per audio file)
    expected_audio_count = len(entries) // 2
    if not verify_audio_files(audio_dir, prefix, expected_audio_count):
        return False
    
    # Step 4: Update JSONL entries
    updated_entries = update_sakura_entries(entries, dataset_name, prefix)
    if not updated_entries:
        return False
    
    # Step 5: Save updated JSONL
    if not save_updated_jsonl(updated_entries, jsonl_path):
        return False
    
    # Step 6: Verify results
    if verify_sakura_results(jsonl_path, dataset_name, prefix):
        print(f"\n🎉 SUCCESS: {dataset_name.upper()} dataset nomenclature fixed!")
        return True
    else:
        print(f"\n❌ FAILED: Issues found during verification for {dataset_name.upper()}")
        return False

def main():
    print("=== SAKURA Dataset Nomenclature Fix Script ===\n")
    
    successful_datasets = []
    failed_datasets = []
    
    for dataset_name, config in SAKURA_DATASETS.items():
        if process_sakura_dataset(dataset_name, config):
            successful_datasets.append(dataset_name)
        else:
            failed_datasets.append(dataset_name)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    if successful_datasets:
        print(f"✓ Successfully processed: {', '.join(successful_datasets)}")
    
    if failed_datasets:
        print(f"❌ Failed to process: {', '.join(failed_datasets)}")
    
    if len(successful_datasets) == len(SAKURA_DATASETS):
        print(f"\n🎉 ALL SAKURA DATASETS PROCESSED SUCCESSFULLY!")
        print("Changes made:")
        print("- Updated audio_path to match existing sequential filenames")
        print("- Removed audio_path_status field")
        print("- Maintained 2 entries per audio file (single + multi hop)")
        print("- Created backups of original JSONL files")
    else:
        print(f"\n⚠️  Some datasets failed to process. Check error messages above.")

if __name__ == "__main__":
    main()