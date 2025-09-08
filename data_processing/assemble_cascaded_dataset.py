# data_processing/assemble_cascaded_dataset.py

import os
import json
import argparse
from pathlib import Path
import sys

# Add the project root to the Python path to allow importing our config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def assemble_dataset(source_dir: str, intermediate_dir: str, output_dir: str):
    """
    This is the final assembly step. It takes the original dataset metadata,
    finds the corresponding ASR and Captioning text files we generated,
    and combines them into a new, final standardized JSONL file for our
    'cascaded_text' experiments.
    """
    source_path = Path(source_dir)
    intermediate_path = Path(intermediate_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the paths to our intermediate results
    asr_path = intermediate_path / "asr"
    caption_path = intermediate_path / "captioning"

    try:
        # We start with the original JSONL file as our "template".
        jsonl_file = next(source_path.glob('*_standardized.jsonl'))
    except StopIteration:
        print(f"ERROR: No original '*_standardized.jsonl' file found in '{source_dir}'. Skipping.")
        return

    print(f"\nAssembling cascaded dataset for: {source_path.name}")
    
    new_jsonl_data = []
    original_data = [json.loads(line) for line in open(jsonl_file, 'r')]

    for item in original_data:
        item_id = item['id']
        # Construct the expected paths for the intermediate text files.
        asr_txt_file = asr_path / f"{item_id}_asr.txt"
        caption_txt_file = caption_path / f"{item_id}_caption.txt"

        # This is a critical robustness check. If either the ASR or Caption file
        # is missing for an ID, we skip it to avoid creating incomplete data.
        if not asr_txt_file.exists() or not caption_txt_file.exists():
            print(f"  - WARNING: Missing ASR or Caption file for ID {item_id}. Skipping.")
            continue

        # Read the content from our intermediate files.
        with open(asr_txt_file, 'r', encoding='utf-8') as f:
            asr_text = f.read().strip()
        with open(caption_txt_file, 'r', encoding='utf-8') as f:
            caption_text = f.read().strip()

        # This is where we construct the final combined text block, using the
        # labeled format we agreed upon for maximum clarity for the model.
        combined_text = (
            f"Audio/Speech Transcription: {asr_text}\n"
            f"Audio/Speech Information: {caption_text}"
        )

        new_item = item.copy()
        # Here, we repurpose the 'audio_path' field to hold our new combined text context.
        # This is the key "hack" that allows us to reuse all our experiment scripts without modification.
        new_item['audio_path'] = combined_text
        new_jsonl_data.append(new_item)

    # Construct the final output filename according to our convention.
    output_jsonl_filename = f"{source_path.name}_cascaded_standardized.jsonl"
    output_jsonl_path = output_path / output_jsonl_filename
    
    with open(output_jsonl_path, 'w') as f:
        for entry in new_jsonl_data:
            # We use ensure_ascii=False to maintain human readability, per our SOP.
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"  - Successfully assembled {len(new_jsonl_data)} items.")
    print(f"  - Final cascaded dataset saved to: {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the final cascaded text dataset from ASR and Captioning outputs.")
    parser.add_argument('--source', type=str, required=True, help="Path to the original source dataset directory (e.g., 'data/sakura').")
    parser.add_argument('--intermediate', type=str, required=True, help="Path to the top-level directory containing the 'asr' and 'captioning' subfolders (e.g., 'data/sakura_cascaded').")
    args = parser.parse_args()

    source_base = Path(args.source)
    intermediate_base = Path(args.intermediate)
    
    # This logic correctly handles both single-directory (MMAR) and multi-directory (Sakura) datasets.
    if source_base.name == 'sakura':
        sub_dirs = [d for d in source_base.iterdir() if d.is_dir() and d.name != 'audio']
        for sub_dir in sub_dirs:
            intermediate_sub_dir = intermediate_base / sub_dir.name
            # The final output goes into the same intermediate directory.
            assemble_dataset(str(sub_dir), str(intermediate_sub_dir), str(intermediate_sub_dir))
    else:
        assemble_dataset(args.source, args.intermediate, args.intermediate)

    print("\nCascaded dataset assembly complete.")