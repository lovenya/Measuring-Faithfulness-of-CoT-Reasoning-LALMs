# data_processing/generate_whisper_asr.py

import os
import json
import argparse
from pathlib import Path
import sys

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# To make sure our project's modules (like 'config') can be found, we add the
# project's root directory to the Python path. This is a robust way to handle imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ==============================================================================
# --- Core Transcription Logic (Adapted from your proven script) ---
# This section contains the essential functions for loading audio and running
# Whisper inference. We've encapsulated your battle-tested logic here to ensure
# it's used consistently and correctly for every audio file.
# ==============================================================================

def load_audio(path, target_sr=16000):
    """
    A helper function to load an audio file from a given path.
    It handles two critical pre-processing steps:
    1. Converts stereo audio to mono by averaging the channels.
    2. Resamples the audio to the 16kHz sample rate that Whisper requires.
    """
    speech, sr = torchaudio.load(path)
    if speech.shape[0] > 1:  # Check if it's stereo
        speech = speech.mean(dim=0, keepdim=True)
    speech = speech.squeeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        speech = resampler(speech)
    return speech.numpy(), target_sr

def forced_decoder_prompt_length(forced_decoder_ids):
    """A small helper to safely calculate the length of the forced decoder IDs."""
    if not forced_decoder_ids: return 0
    try:
        if any(isinstance(x, (list, tuple)) for x in forced_decoder_ids):
            return sum(len(x) for x in forced_decoder_ids)
        return len(forced_decoder_ids)
    except Exception:
        return 0

def transcribe_audio_file(model, processor, audio_path, device, requested_max_new_tokens, num_beams):
    """
    This is the main workhorse function, built from your proven script. It takes a
    single audio file, processes it, and returns the transcribed text.
    """
    # First, we load and prepare the audio data into the correct format.
    speech_array, sampling_rate = load_audio(audio_path, target_sr=16000)
    
    # The processor converts the raw audio waveform into the 'features' the model expects.
    # We ensure these features are on the GPU and have the same data type (e.g., float16)
    # as the model to prevent any mismatches.
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device).to(model.dtype)
    
    # The attention mask tells the model which parts of the input to pay attention to.
    # For audio, we want it to consider the entire input.
    attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long, device=device)
    
    # This is a key step for accuracy: we force the model to perform English transcription.
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    
    # --- The Critical Fix for the Max Length Error ---
    # This is the robust logic from your script that calculates the maximum number of
    # new tokens we can safely generate without crashing the model. It accounts for the
    # length of the forced prompt and leaves a safety buffer, preventing the off-by-one
    # error we previously encountered.
    max_target_positions = getattr(model.config, "max_target_positions", 448)
    decoder_prompt_len = forced_decoder_prompt_length(forced_decoder_ids)
    available_for_generation = max_target_positions - decoder_prompt_len
    if available_for_generation <= 1:
        raise RuntimeError("Not enough room for generation after setting forced decoder IDs.")
    safe_max_new_tokens = min(requested_max_new_tokens, max(1, available_for_generation - 1))
    
    # We bundle all the generation parameters together for the model.
    gen_kwargs = dict(max_new_tokens=safe_max_new_tokens, num_beams=num_beams)
    if forced_decoder_ids is not None:
        gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

    # We run the actual inference within a 'no_grad' block, a standard PyTorch
    # optimization that speeds up the process by not calculating gradients.
    with torch.no_grad():
        generated_ids = model.generate(input_features=input_features, attention_mask=attention_mask, **gen_kwargs)
        
    # Finally, we decode the model's numerical output (token IDs) back into human-readable text.
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.strip()

# ==============================================================================
# --- Dataset Processing Framework ---
# This section contains our standard, robust logic for iterating through our
# datasets, processing each file, and handling errors gracefully.
# ==============================================================================

def process_dataset_for_asr(model, processor, source_dir: str, output_dir: str, device, max_new_tokens, num_beams):
    """
    Processes a single dataset directory (e.g., 'data/sakura/animal'), running
    Whisper ASR on each audio file and saving the transcription to a dedicated text file.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    # We create a dedicated 'asr' subfolder to keep our output files organized.
    asr_output_path = output_path / "asr"
    asr_output_path.mkdir(parents=True, exist_ok=True)

    try:
        # We need the original JSONL file to know which audio files belong to the dataset and to get their unique IDs.
        jsonl_file = next(source_path.glob('*_standardized.jsonl'))
    except StopIteration:
        print(f"ERROR: No '*_standardized.jsonl' file found in '{source_dir}'. Skipping.")
        return

    print(f"\nProcessing ASR for dataset: {source_path.name}")
    print(f"  - Outputting ASR .txt files to: {asr_output_path}")

    original_data = [json.loads(line) for line in open(jsonl_file, 'r')]

    for i, item in enumerate(original_data):
        audio_file = Path(item['audio_path'])
        item_id = item['id']
        output_txt_path = asr_output_path / f"{item_id}_asr.txt"

        if not audio_file.exists():
            print(f"  - WARNING: Source audio file not found, skipping: {audio_file}")
            continue
        
        # This is a simple but effective optimization: if the output file already exists,
        # we assume it's been processed and skip it. This makes the script resumable.
        if output_txt_path.exists():
            continue

        if config.VERBOSE:
            print(f"  - Transcribing (ASR) {i+1}/{len(original_data)}: {audio_file.name}")
        try:
            # Here, we call our main workhorse function to get the transcription.
            transcription = transcribe_audio_file(model, processor, str(audio_file), device, max_new_tokens, num_beams)
            # We save the result to a plain text file, encoded in UTF-8 to handle any special characters.
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
        except Exception as e:
            # This robust error handling ensures that if one file fails, the whole job doesn't crash.
            print(f"  - ERROR during Whisper transcription for {audio_file.name}: {e}")
            continue

if __name__ == "__main__":
    # --- Command-Line Interface Setup ---
    parser = argparse.ArgumentParser(description="Generate ASR transcriptions using a local Whisper model.")
    parser.add_argument('--source', type=str, required=True, help="Path to the source dataset directory (e.g., 'data/sakura').")
    parser.add_argument('--output', type=str, required=True, help="Path to the top-level output directory for the cascaded dataset (e.g., 'data/sakura_cascaded').")
    parser.add_argument("--max_new_tokens", type=int, default=448)
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    # --- Model and Device Setup ---
    # We hard-code the device to 'cuda' for simplicity, as this script is
    # intended for our GPU-enabled HPC environment.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Running on CPU. This will be extremely slow.")

    # We load the model path from our central config file, adhering to our
    # "single source of truth" principle.
    model_path = config.WHISPER_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"FATAL: Whisper model directory not found at path specified in config.py: '{model_path}'")
        sys.exit(1)

    print(f"Loading Whisper model from: {model_path}...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval() # Set the model to evaluation mode, a standard practice for inference.
    print("Whisper model loaded successfully.")

    # --- Dataset Iteration Logic ---
    # This logic allows the script to be flexible, handling both single-directory
    # datasets (like MMAR) and multi-directory datasets (like Sakura) gracefully.
    source_base = Path(args.source)
    output_base = Path(args.output)
    
    if source_base.name == 'sakura':
        # If the user points to the main 'sakura' folder, we're smart enough to find
        # and process all its sub-tracks (animal, emotion, etc.) automatically.
        sub_dirs = [d for d in source_base.iterdir() if d.is_dir() and d.name != 'audio']
        for sub_dir in sub_dirs:
            output_sub_dir = output_base / sub_dir.name
            process_dataset_for_asr(model, processor, str(sub_dir), str(output_sub_dir), device, args.max_new_tokens, args.num_beams)
    else:
        # Otherwise, we just process the single directory the user specified.
        process_dataset_for_asr(model, processor, args.source, args.output, device, args.max_new_tokens, args.num_beams)

    print("\nWhisper ASR generation complete.")