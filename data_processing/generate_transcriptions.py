# data_processing/generate_transcriptions.py

import os
import json
import argparse
from pathlib import Path
import sys

# Add the project root to the Python path to allow importing our core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.lalm_utils import load_model_and_tokenizer, run_inference

# This is the prompt we designed to elicit both transcription and description.
TRANSCRIPTION_PROMPT = "Listen to the following audio carefully and provide a detailed and accurate description. Transcribe any spoken words verbatim. Also, describe any other significant sounds, such as music, environmental noises, or animal sounds, and note their relationship to the speech if relevant."

def process_single_dataset(model, processor, source_dir: str, output_dir: str):
    """
    Processes a single dataset directory (e.g., 'data/sakura/animal').
    It loads each audio file, generates a transcription/description using the LALM,
    and creates a new, text-only standardized JSONL file.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    try:
        # Find the original standardized JSONL file in the source directory.
        jsonl_file = next(source_path.glob('*_standardized.jsonl'))
    except StopIteration:
        print(f"ERROR: No '*_standardized.jsonl' file found in '{source_dir}'. Skipping.")
        return

    print(f"\nProcessing dataset: {source_path.name}")
    print(f"  - Source JSONL: {jsonl_file.name}")
    print(f"  - Output directory: {output_path}")

    # Create the output directory if it doesn't exist.
    output_path.mkdir(parents=True, exist_ok=True)

    new_jsonl_data = []
    original_data = [json.loads(line) for line in open(jsonl_file, 'r')]

    for i, item in enumerate(original_data):
        clean_audio_path = Path(item['audio_path'])
        
        if not clean_audio_path.exists():
            print(f"  - WARNING: Source audio file not found, skipping: {clean_audio_path}")
            continue
            
        print(f"  - Transcribing audio {i+1}/{len(original_data)}: {clean_audio_path.name}")

        try:
            # --- Core Transcription Logic ---
            messages = [{"role": "user", "content": TRANSCRIPTION_PROMPT}]
            
            # We use do_sample=False to get the single most likely, deterministic description.
            transcribed_text = run_inference(
                model, processor, messages, str(clean_audio_path), max_new_tokens=512, do_sample=False
            )

            # --- Repurposing the JSON structure ---
            new_item = item.copy()
            # This is the key step: we replace the file path with the generated text.
            new_item['audio_path'] = transcribed_text
            # We add a new key for clarity and remove the original path.
            new_item['transcription_source'] = 'qwen2-audio-7b'
            if 'original_audio_path' in new_item: del new_item['original_audio_path']
            
            new_jsonl_data.append(new_item)

        except Exception as e:
            print(f"  - ERROR processing {clean_audio_path.name}: {e}. Skipping.")
            continue

    # Construct the new filename according to our convention.
    output_jsonl_filename = f"{source_path.name}_transcribed_audio_standardized.jsonl"
    output_jsonl_path = output_path / output_jsonl_filename
    
    with open(output_jsonl_path, 'w') as f:
        for entry in new_jsonl_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"  - Successfully generated {len(new_jsonl_data)} transcriptions.")
    print(f"  - New JSONL file created at: {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate transcribed/described text for an audio dataset using the Qwen-Audio model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--source', type=str, required=True, help="Path to the source dataset directory (e.g., 'data/sakura/animal' or 'data/sakura').")
    parser.add_argument('--output', type=str, required=True, help="Path to the output directory for the new transcribed dataset (e.g., 'data/sakura_transcribed/animal').")
    args = parser.parse_args()

    # --- Load the Model Once ---
    # This is a heavyweight operation, so we do it once at the start.
    print("Loading Qwen-Audio model for transcription...")
    model, processor = load_model_and_tokenizer(config.MODEL_PATH)
    print("Model loaded.")

    # --- Logic to handle multiple source directories (for Sakura) ---
    source_base = Path(args.source)
    output_base = Path(args.output)
    
    if source_base.name == 'sakura':
        # If the source is the main 'sakura' folder, find and process all sub-tracks.
        sub_dirs = [d for d in source_base.iterdir() if d.is_dir() and d.name != 'audio']
        for sub_dir in sub_dirs:
            output_sub_dir = output_base / sub_dir.name
            process_single_dataset(model, processor, str(sub_dir), str(output_sub_dir))
    else:
        # Otherwise, process the single specified directory.
        process_single_dataset(model, processor, args.source, args.output)

    print("\nTranscription generation complete.")