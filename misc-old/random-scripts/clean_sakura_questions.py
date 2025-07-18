# utils/clean_sakura_questions.py

import json
import os
import re
from pathlib import Path

def clean_question_text(question: str) -> str:
    """
    Removes embedded multiple-choice options from a question string.
    This function finds the first occurrence of a pattern like '(a)' or '(b)'
    and truncates the string just before it, removing the choices.

    Args:
        question (str): The original question string.

    Returns:
        str: The cleaned question string.
    """
    # This regex looks for a space, followed by '(', a letter, and ')'
    # This is a reliable indicator of the start of the choices list.
    match = re.search(r'\s*\([a-z]\)', question)
    if match:
        # Return the part of the string before the match
        return question[:match.start()].strip()
    
    # If no match is found, return the original question
    return question

def process_file(input_path: Path, output_path: Path):
    """
    Reads a JSONL file, cleans the 'question' field for each line,
    and writes the result to a new JSONL file.
    """
    print(f"Processing '{input_path}'...")
    cleaned_lines = 0
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                
                original_question = data.get('question', '')
                cleaned_question = clean_question_text(original_question)
                
                if original_question != cleaned_question:
                    cleaned_lines += 1
                
                data['question'] = cleaned_question
                
                outfile.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError:
                print(f"  - Warning: Skipping malformed JSON line.")
                outfile.write(line) # Write the original line if it can't be parsed

    print(f"  - Done. Cleaned {cleaned_lines} questions.")
    print(f"  - Saved cleaned file to '{output_path}'")

def main():
    """
    Main function to define file paths and run the cleaning process.
    """
    # Assumes the script is run from the project root directory
    # 'MEASURING-FAITHFULNESS-OF-COT-REASONING-LALMS/'
    project_root = Path.cwd()
    data_dir = project_root / 'data' / 'sakura'
    
    # Define the four Sakura dataset files to be cleaned
    sakura_files = [
        "animal/sakura_animal_test_standardized.jsonl",
        "emotion/sakura_emotion_test_standardized.jsonl",
        "gender/sakura_gender_test_standardized.jsonl",
        "language/sakura_language_test_standardized.jsonl"
    ]

    print("--- Starting Sakura Dataset Question Cleaning ---")
    for filename in sakura_files:
        input_file = data_dir / filename
        # Save the cleaned file with a '_cleaned' suffix
        output_file = data_dir / f"{input_file.stem}_cleaned.jsonl"
        
        if input_file.exists():
            process_file(input_file, output_file)
        else:
            print(f"Warning: File not found, skipping: '{input_file}'")
    print("--- Cleaning process complete. ---")
    print("\nNext steps:")
    print("1. Manually inspect the new '_cleaned.jsonl' files to verify the changes.")
    print("2. If they look correct, back up the original files.")
    print("3. Rename the '_cleaned.jsonl' files to their original names.")
    print("   Example: mv data/sakura/sakura_animal_test_standardized_cleaned.jsonl data/sakura/sakura_animal_test_standardized.jsonl")

if __name__ == "__main__":
    main()