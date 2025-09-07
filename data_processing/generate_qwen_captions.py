# data_processing/generate_qwen_captions.py

import os
import json
import argparse
from pathlib import Path
import sys

# Add the project root to the Python path to allow importing our core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.lalm_utils import load_model_and_tokenizer, run_inference

# This prompt is now focused specifically on captioning non-speech sounds,
# as the ASR component is handled separately by Whisper.
CAPTIONING_PROMPT = "Describe only the non-speech sounds in this audio, such as music, environmental noises, or animal sounds. Ignore any spoken words. If there are no non-speech sounds, respond with 'No non-speech sounds detected.'"

def process_dataset_for_captioning(model, processor, source_dir: str, output_dir: str):
    """
    Processes a single dataset directory, running Qwen-Audio to generate a caption
    for each audio file and saving it to a dedicated text file.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    # The output is now saved to a 'captioning' subfolder for clear organization.
    caption_output_path = output_path / "captioning"
    caption_output_path.mkdir(parents=True, exist_ok=True)

    try:
        jsonl_file = next(source_path.glob('*_standardized.jsonl'))
    except StopIteration:
        print(f"ERROR: No '*_standardized.jsonl' file found in '{source_dir}'. Skipping.")
        return

    print(f"\nProcessing Audio Captioning for dataset: {source_path.name}")
    print(f"  - Outputting Caption .txt files to: {caption_output_path}")

    original_data = [json.loads(line) for line in open(jsonl_file, 'r')]

    for i, item in enumerate(original_data):
        audio_file = Path(item['audio_path'])
        item_id = item['id']
        # The output filename is now standardized to match our assembly script's expectations.
        output_txt_path = caption_output_path / f"{item_id}_caption.txt"

        if not audio_file.exists():
            print(f"  - WARNING: Source audio file not found, skipping: {audio_file}")
            continue
        
        # Skip if the output file already exists to make the script resumable.
        if output_txt_path.exists():
            continue

        if config.VERBOSE:
            print(f"  - Captioning (AAC) {i+1}/{len(original_data)}: {audio_file.name}")
        try:
            messages = [{"role": "user", "content": CAPTIONING_PROMPT}]
            # We use deterministic inference to get the single most likely caption.
            caption_text = run_inference(
                model, processor, messages, str(audio_file), max_new_tokens=256, do_sample=False
            )
            # Save the raw text output to the file.
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(caption_text)
        except Exception as e:
            print(f"  - ERROR during Qwen captioning for {audio_file.name}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio captions using the Qwen-Audio model.")
    parser.add_argument('--source', type=str, required=True, help="Path to the source dataset directory (e.g., 'data/sakura').")
    parser.add_argument('--output', type=str, required=True, help="Path to the top-level output directory for the cascaded dataset (e.g., 'data/sakura_cascaded').")
    args = parser.parse_args()

    # --- Load the Model Once ---
    print("Loading Qwen-Audio model for captioning...")
    model, processor = load_model_and_tokenizer(config.MODEL_PATH)
    print("Model loaded.")

    # --- Dataset Iteration Logic ---
    source_base = Path(args.source)
    output_base = Path(args.output)
    
    if source_base.name == 'sakura':
        sub_dirs = [d for d in source_base.iterdir() if d.is_dir() and d.name != 'audio']
        for sub_dir in sub_dirs:
            output_sub_dir = output_base / sub_dir.name
            process_dataset_for_captioning(model, processor, str(sub_dir), str(output_sub_dir))
    else:
        process_dataset_for_captioning(model, processor, args.source, args.output)

    print("\nQwen Audio Captioning generation complete.")