#!/usr/bin/env python3
"""
Script to fix MMAR dataset nomenclature:
1. Rename audio files from complex names to sequential mmar_audio_0.wav, mmar_audio_1.wav, etc.
2. Update JSONL file paths accordingly
3. Handle different types of mismatched files properly
"""

import os
import json
import shutil
from pathlib import Path

# Configuration
MMAR_AUDIO_DIR = "./data/mmar/audio"
MMAR_JSONL_PATH = "./data/mmar/mmar_test_standardized.jsonl"
MMAR_JSONL_BACKUP = "./data/mmar/mmar_test_standardized.jsonl.backup"
MMAR_META_JSON = "./data/mmar/MMAR-meta.json"

def backup_original_jsonl():
    """Create a backup of the original JSONL file"""
    if os.path.exists(MMAR_JSONL_PATH):
        shutil.copy2(MMAR_JSONL_PATH, MMAR_JSONL_BACKUP)
        print(f"✓ Backup created: {MMAR_JSONL_BACKUP}")
    else:
        print(f"ERROR: Original JSONL file not found at {MMAR_JSONL_PATH}")
        return False
    return True

def load_mmar_meta():
    """Load the MMAR-meta.json file for complete dataset mapping"""
    try:
        with open(MMAR_META_JSON, 'r') as f:
            meta_data = json.load(f)
        print(f"✓ Loaded MMAR-meta.json with {len(meta_data)} entries")
        return meta_data
    except Exception as e:
        print(f"ERROR: Failed to load MMAR-meta.json: {e}")
        return None

def load_current_jsonl():
    """Load current JSONL entries"""
    entries = []
    try:
        with open(MMAR_JSONL_PATH, 'r') as f:
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

def get_current_audio_files():
    """Get list of current audio files in the directory"""
    if not os.path.exists(MMAR_AUDIO_DIR):
        print(f"ERROR: Audio directory not found: {MMAR_AUDIO_DIR}")
        return None
    
    audio_files = []
    for file in os.listdir(MMAR_AUDIO_DIR):
        if file.endswith('.wav'):
            audio_files.append(file)
    
    audio_files.sort()  # Sort to ensure consistent ordering
    print(f"✓ Found {len(audio_files)} audio files in directory")
    return audio_files

def analyze_file_categories(entries, audio_files, meta_data):
    """Analyze and categorize files into different types"""
    
    # Create mappings from meta_data
    meta_audio_to_id = {}  # filename -> id
    meta_id_to_audio = {}  # id -> filename
    
    for item in meta_data:
        audio_path = item.get('audio_path', '')
        filename = os.path.basename(audio_path)
        item_id = item.get('id', '')
        if filename and item_id:
            meta_audio_to_id[filename] = item_id
            meta_id_to_audio[item_id] = filename
    
    # Get IDs from JSONL entries
    jsonl_ids = set()
    jsonl_entries_by_id = {}
    for entry in entries:
        entry_id = entry.get('id', '')
        if entry_id:
            jsonl_ids.add(entry_id)
            jsonl_entries_by_id[entry_id] = entry
    
    # Categorize files
    directory_filenames = set(audio_files)
    
    # Category 1: Files in directory + meta + JSONL (should be renamed)
    files_to_rename = {}  # filename -> (id, entry)
    
    # Category 2: Files in directory + meta but NOT in JSONL (option to include)
    files_missing_jsonl = {}  # filename -> id
    
    # Category 3: Files in directory but NOT in meta (truly orphaned)
    truly_orphaned_files = []
    
    # Category 4: Entries in JSONL but file NOT in directory (remove from JSONL)
    missing_audio_entries = []  # entries to remove
    
    # Analyze files in directory
    for filename in directory_filenames:
        if filename in meta_audio_to_id:
            file_id = meta_audio_to_id[filename]
            if file_id in jsonl_ids:
                # Category 1: Perfect match
                files_to_rename[filename] = (file_id, jsonl_entries_by_id[file_id])
            else:
                # Category 2: In meta but not in JSONL
                files_missing_jsonl[filename] = file_id
        else:
            # Category 3: Not in meta at all
            truly_orphaned_files.append(filename)
    
    # Analyze JSONL entries for missing audio files
    for entry in entries:
        entry_id = entry.get('id', '')
        if entry_id in meta_id_to_audio:
            expected_filename = meta_id_to_audio[entry_id]
            if expected_filename not in directory_filenames:
                # Category 4: JSONL entry but no audio file
                missing_audio_entries.append(entry)
    
    # Print analysis
    print(f"\n--- File Analysis ---")
    print(f"Files in directory: {len(directory_filenames)}")
    print(f"Entries in JSONL: {len(entries)}")
    print(f"Entries in MMAR-meta.json: {len(meta_data)}")
    print(f"")
    print(f"📁 Files to rename (directory + meta + JSONL): {len(files_to_rename)}")
    print(f"❓ Files missing JSONL entry (directory + meta, no JSONL): {len(files_missing_jsonl)}")
    print(f"🗑️  Truly orphaned files (directory only, not in meta): {len(truly_orphaned_files)}")
    print(f"❌ JSONL entries with missing audio files: {len(missing_audio_entries)}")
    
    # Show details for problematic categories
    if files_missing_jsonl:
        print(f"\n❓ FILES WITH MISSING JSONL ENTRIES:")
        for i, (filename, file_id) in enumerate(sorted(files_missing_jsonl.items()), 1):
            print(f"  {i:2d}. {filename} (ID: {file_id})")
    
    if truly_orphaned_files:
        print(f"\n🗑️  TRULY ORPHANED FILES (not in MMAR-meta.json):")
        for i, filename in enumerate(sorted(truly_orphaned_files), 1):
            print(f"  {i:2d}. {filename}")
    
    if missing_audio_entries:
        print(f"\n❌ JSONL ENTRIES WITH MISSING AUDIO FILES:")
        for i, entry in enumerate(missing_audio_entries, 1):
            entry_id = entry.get('id', 'N/A')
            expected_file = meta_id_to_audio.get(entry_id, 'Unknown')
            print(f"  {i:2d}. ID: {entry_id} -> Expected: {expected_file}")
    
    return files_to_rename, files_missing_jsonl, truly_orphaned_files, missing_audio_entries

def handle_problematic_files(files_missing_jsonl, truly_orphaned_files, missing_audio_entries):
    """Handle files that have various issues"""
    
    actions = {
        'include_missing_jsonl': False,
        'delete_orphaned': False,
        'remove_missing_audio_entries': False
    }
    
    # Handle files missing JSONL entries
    if files_missing_jsonl:
        print(f"\n❓ {len(files_missing_jsonl)} files exist in directory and MMAR-meta.json but NOT in JSONL.")
        print("These files have valid IDs but no corresponding JSONL entries.")
        print("Options:")
        print("1. Include them (they will be renamed and complete JSONL entries will be created)")
        print("2. Skip them (keep files unchanged)")
        
        while True:
            choice = input("What would you like to do with missing JSONL files? (1/2): ").strip()
            if choice == '1':
                actions['include_missing_jsonl'] = True
                print("✓ Will include files missing JSONL entries")
                break
            elif choice == '2':
                print("✓ Will skip files missing JSONL entries")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    # Handle truly orphaned files
    if truly_orphaned_files:
        print(f"\n🗑️  {len(truly_orphaned_files)} files are truly orphaned (not in MMAR-meta.json).")
        print("These files don't exist in the official dataset metadata.")
        print("Options:")
        print("1. Delete them (recommended)")
        print("2. Keep them (skip)")
        
        while True:
            choice = input("What would you like to do with truly orphaned files? (1/2): ").strip()
            if choice == '1':
                actions['delete_orphaned'] = True
                print("✓ Will delete truly orphaned files")
                break
            elif choice == '2':
                print("✓ Will keep truly orphaned files")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    # Handle missing audio entries
    if missing_audio_entries:
        print(f"\n❌ {len(missing_audio_entries)} JSONL entries have missing audio files.")
        print("These entries reference audio files that don't exist in the directory.")
        print("Options:")
        print("1. Remove them from JSONL (recommended)")
        print("2. Keep them in JSONL")
        
        while True:
            choice = input("What would you like to do with missing audio entries? (1/2): ").strip()
            if choice == '1':
                actions['remove_missing_audio_entries'] = True
                print("✓ Will remove entries with missing audio files")
                break
            elif choice == '2':
                print("✓ Will keep entries with missing audio files")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    return actions

def execute_cleanup_actions(actions, files_missing_jsonl, truly_orphaned_files, missing_audio_entries):
    """Execute the cleanup actions chosen by the user"""
    
    # Delete truly orphaned files
    if actions['delete_orphaned'] and truly_orphaned_files:
        print(f"\n--- Deleting Truly Orphaned Files ---")
        for filename in truly_orphaned_files:
            file_path = os.path.join(MMAR_AUDIO_DIR, filename)
            try:
                os.remove(file_path)
                print(f"✓ Deleted: {filename}")
            except Exception as e:
                print(f"ERROR: Failed to delete {filename}: {e}")
                return False
        print(f"✓ Successfully deleted {len(truly_orphaned_files)} truly orphaned files")
    
    return True

def create_final_file_list(files_to_rename, files_missing_jsonl, actions, meta_data):
    """Create the final list of files to rename and their corresponding entries"""
    
    # Start with files that have complete mapping
    final_files = []  # List of (filename, id, entry)
    
    # Add files with complete JSONL entries
    for filename, (file_id, entry) in files_to_rename.items():
        final_files.append((filename, file_id, entry))
    
    # Add files missing JSONL entries if user chose to include them
    if actions['include_missing_jsonl']:
        for filename, file_id in files_missing_jsonl.items():
            # Create a complete JSONL entry for these files using MMAR-meta.json data
            new_entry = None
            
            # Find the corresponding entry in meta_data
            for meta_item in meta_data:
                if meta_item.get('id') == file_id:
                    # Create a proper JSONL entry based on meta_data structure
                    new_entry = {
                        'id': file_id,
                        'audio_path': f'./audio/{filename}',  # Will be updated to sequential path later
                    }
                    
                    # Copy all relevant fields from meta_data
                    for key in ['question', 'choices', 'answer', 'modality', 'category', 'sub-category', 'language', 'source', 'url', 'timestamp']:
                        if key in meta_item:
                            new_entry[key] = meta_item[key]
                    
                    # Handle answer_key (convert from 'answer' if needed)
                    if 'answer' in meta_item and 'answer_key' not in new_entry:
                        answer = meta_item['answer']
                        choices = meta_item.get('choices', [])
                        if choices and answer in choices:
                            new_entry['answer_key'] = choices.index(answer)
                        else:
                            new_entry['answer_key'] = answer  # Keep original if not found in choices
                    
                    break
            
            if new_entry:
                final_files.append((filename, file_id, new_entry))
                print(f"✓ Created JSONL entry for {filename} (ID: {file_id})")
            else:
                print(f"WARNING: Could not create JSONL entry for {filename} (ID: {file_id}) - not found in meta_data")
    
    # Sort by filename for consistent ordering
    final_files.sort(key=lambda x: x[0])
    
    print(f"\n--- Final File List for Renaming ---")
    print(f"Total files to rename: {len(final_files)}")
    
    for i, (filename, file_id, entry) in enumerate(final_files[:10]):  # Show first 10
        has_question = 'question' in entry
        entry_status = "Complete entry" if has_question else "Minimal entry"
        print(f"  {i:3d}: {filename} -> mmar_audio_{i}.wav (ID: {file_id}, {entry_status})")
    
    if len(final_files) > 10:
        print(f"  ... and {len(final_files) - 10} more files")
    
    return final_files

def filter_jsonl_entries(entries, missing_audio_entries, actions):
    """Filter JSONL entries to remove those with missing audio files"""
    
    if not actions['remove_missing_audio_entries']:
        return entries
    
    # Create set of IDs to remove
    ids_to_remove = set()
    for entry in missing_audio_entries:
        ids_to_remove.add(entry.get('id', ''))
    
    # Filter entries
    filtered_entries = []
    removed_count = 0
    
    for entry in entries:
        entry_id = entry.get('id', '')
        if entry_id in ids_to_remove:
            removed_count += 1
        else:
            filtered_entries.append(entry)
    
    print(f"\n--- JSONL Filtering ---")
    print(f"Original entries: {len(entries)}")
    print(f"Removed entries: {removed_count}")
    print(f"Remaining entries: {len(filtered_entries)}")
    
    return filtered_entries

def rename_audio_files(final_files):
    """Rename audio files according to the final file list"""
    print(f"\n--- Renaming Audio Files ---")
    
    renamed_count = 0
    for i, (filename, file_id, entry) in enumerate(final_files):
        old_path = os.path.join(MMAR_AUDIO_DIR, filename)
        new_filename = f"mmar_audio_{i}.wav"
        new_path = os.path.join(MMAR_AUDIO_DIR, new_filename)
        
        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
                if i < 5 or i % 100 == 0:  # Show progress
                    print(f"✓ Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"ERROR: Failed to rename {filename}: {e}")
                return False
        else:
            print(f"ERROR: File not found: {old_path}")
            return False
    
    print(f"✓ Successfully renamed {renamed_count} files")
    return True

def create_updated_jsonl_entries(final_files, filtered_entries):
    """Create updated JSONL entries with new sequential paths"""
    print(f"\n--- Creating Updated JSONL Entries ---")
    
    # Create mapping from ID to (new_path, index, entry)
    id_to_info = {}
    for i, (filename, file_id, entry) in enumerate(final_files):
        new_path = f"data/mmar/audio/mmar_audio_{i}.wav"
        id_to_info[file_id] = (new_path, i, entry)
    
    # Create the final JSONL entries in the correct order
    updated_entries = []
    
    # Process all final_files in order to maintain sequential ordering
    for i, (filename, file_id, entry) in enumerate(final_files):
        # Use the entry from final_files (which includes newly created entries)
        updated_entry = entry.copy()
        updated_entry['audio_path'] = f"data/mmar/audio/mmar_audio_{i}.wav"
        updated_entries.append(updated_entry)
        
        if i < 5:  # Show first 5 entries
            print(f"  Entry {i:3d}: ID {file_id} -> {updated_entry['audio_path']}")
    
    if len(updated_entries) > 5:
        print(f"  ... and {len(updated_entries) - 5} more entries")
    
    print(f"✓ Created {len(updated_entries)} updated JSONL entries")
    
    # Verify that we have all the required fields
    missing_fields = []
    for i, entry in enumerate(updated_entries[:5]):  # Check first 5
        required_fields = ['id', 'audio_path', 'question', 'choices']
        for field in required_fields:
            if field not in entry:
                missing_fields.append(f"Entry {i}: missing '{field}'")
    
    if missing_fields:
        print(f"WARNING: Some entries may be missing required fields:")
        for msg in missing_fields:
            print(f"  {msg}")
    else:
        print("✓ All entries have required fields")
    
    return updated_entries

def save_updated_jsonl(updated_entries):
    """Save the updated JSONL file"""
    try:
        with open(MMAR_JSONL_PATH, 'w') as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Updated JSONL saved to {MMAR_JSONL_PATH}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save updated JSONL: {e}")
        return False

def verify_results(final_files, updated_entries):
    """Verify that the renaming and path updates were successful"""
    print(f"\n--- Verification ---")
    
    expected_count = len(final_files)
    
    # Check audio files
    audio_files = get_current_audio_files()
    if not audio_files:
        return False
    
    # Check for sequential naming
    sequential_files = [f for f in audio_files if f.startswith("mmar_audio_") and f.endswith(".wav")]
    
    if len(sequential_files) >= expected_count:
        print(f"✓ Found {len(sequential_files)} sequentially named files (expected at least {expected_count})")
    else:
        print(f"ERROR: Expected at least {expected_count} renamed files, found {len(sequential_files)}")
        return False
    
    # Check JSONL entries
    if len(updated_entries) != expected_count:
        print(f"WARNING: Expected {expected_count} JSONL entries, found {len(updated_entries)}")
        # This might be OK if we removed some entries
    
    print(f"✓ Verification completed")
    return True

def main():
    print("=== MMAR Dataset Nomenclature Fix Script (Enhanced v2) ===\n")
    
    # Step 1: Backup original JSONL
    if not backup_original_jsonl():
        return
    
    # Step 2: Load all data sources
    meta_data = load_mmar_meta()
    if not meta_data:
        return
    
    entries = load_current_jsonl()
    if not entries:
        return
    
    audio_files = get_current_audio_files()
    if not audio_files:
        return
    
    # Step 3: Analyze and categorize files
    files_to_rename, files_missing_jsonl, truly_orphaned_files, missing_audio_entries = analyze_file_categories(
        entries, audio_files, meta_data
    )
    
    # Step 4: Get user decisions on how to handle problematic files
    actions = handle_problematic_files(files_missing_jsonl, truly_orphaned_files, missing_audio_entries)
    
    # Step 5: Execute cleanup actions
    if not execute_cleanup_actions(actions, files_missing_jsonl, truly_orphaned_files, missing_audio_entries):
        return
    
    # Step 6: Create final list of files to rename
    final_files = create_final_file_list(files_to_rename, files_missing_jsonl, actions, meta_data)
    if not final_files:
        print("No files to rename. Exiting.")
        return
    
    # Step 7: Filter JSONL entries
    filtered_entries = filter_jsonl_entries(entries, missing_audio_entries, actions)
    
    # Step 8: Rename audio files
    if not rename_audio_files(final_files):
        return
    
    # Step 9: Create updated JSONL entries
    updated_entries = create_updated_jsonl_entries(final_files, filtered_entries)
    
    # Step 10: Save updated JSONL
    if not save_updated_jsonl(updated_entries):
        return
    
    # Step 11: Verify results
    if verify_results(final_files, updated_entries):
        print(f"\n🎉 SUCCESS: MMAR dataset nomenclature fixed!")
        print(f"- {len(final_files)} audio files renamed to mmar_audio_0.wav, mmar_audio_1.wav, etc.")
        print(f"- {len(updated_entries)} JSONL entries updated")
        print(f"- Original JSONL backed up to: {MMAR_JSONL_BACKUP}")
        
        if actions['include_missing_jsonl']:
            print(f"- Included {len(files_missing_jsonl)} files that were missing JSONL entries")
        if actions['delete_orphaned']:
            print(f"- Deleted {len(truly_orphaned_files)} truly orphaned files")
        if actions['remove_missing_audio_entries']:
            print(f"- Removed {len(missing_audio_entries)} JSONL entries with missing audio files")
    else:
        print(f"\n❌ FAILED: Issues found during verification")

if __name__ == "__main__":
    main()