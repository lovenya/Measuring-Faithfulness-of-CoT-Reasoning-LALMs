# fix_sakura_paths.py

import os
import json
import shutil

def fix_audio_paths_in_jsonl(filepath: str):
    """
    Reads a JSONL file, prepends 'data/' to the 'audio_path' field if missing,
    and overwrites the original file with the corrected content.

    Creates a backup of the original file before making any changes.
    """
    print(f"\n--- Processing file: {filepath} ---")

    if not os.path.exists(filepath):
        print(f"  [ERROR] File not found. Skipping.")
        return

    # 1. Safety First: Create a backup of the original file
    backup_path = filepath + '.backup'
    if not os.path.exists(backup_path):
        try:
            shutil.copy(filepath, backup_path)
            print(f"  - Successfully created backup: {backup_path}")
        except Exception as e:
            print(f"  [FATAL] Could not create backup file. Aborting for safety. Error: {e}")
            return
    else:
        print(f"  - Backup file already exists at: {backup_path}")

    # 2. Read file, modify lines in memory
    modified_lines = []
    lines_changed_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 3. Idempotency Check: Only modify if the prefix is missing
                    if 'audio_path' in data and not data['audio_path'].startswith('data/'):
                        # Use os.path.join for robust path construction
                        original_path = data['audio_path']
                        data['audio_path'] = os.path.join('data', original_path)
                        lines_changed_count += 1
                    
                    modified_lines.append(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    print(f"  [WARNING] Could not parse JSON on line {i}. Appending line as-is.")
                    modified_lines.append(line)
                    
    except Exception as e:
        print(f"  [FATAL] Error reading file. Aborting. Error: {e}")
        return

    # 4. Write the modified content back to the original file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        print(f"  - Successfully updated file. {lines_changed_count} line(s) modified.")
    except Exception as e:
        print(f"  [FATAL] Error writing to file. Restore from backup. Error: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # Based on your project structure. Please verify these paths are correct.
    # I have inferred the filenames for emotion, gender, and language.
    SAKURA_TRACKS_FILES = [
        "data/sakura/animal/sakura_animal_test_standardized.jsonl",
        "data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
        "data/sakura/gender/sakura_gender_test_standardized.jsonl",
        "data/sakura/language/sakura_language_test_standardized.jsonl"
    ]
    
    print("Starting Sakura audio_path correction script.")
    for file_to_fix in SAKURA_TRACKS_FILES:
        fix_audio_paths_in_jsonl(file_to_fix)
    print("\n--- Script finished. ---")