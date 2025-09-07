# data_processing/generate_spoken_reasoning.py

import os
import json
import argparse
import torch
import torchaudio
import time
import multiprocessing as mp
import numpy as np # We need numpy for the data type conversion

# This script is designed to be run in a dedicated Conda environment.
# We perform this import check first to give the user a clear error message
# if they are in the wrong environment, saving them from cryptic downstream errors.
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils import io
except ImportError:
    print("FATAL ERROR: The 'TTS' library was not found.")
    print("Please ensure you have activated the dedicated 'tts-env' Conda environment before running this script.")
    exit(1)

# This is a critical "monkey patch" based on our successful test script.
# Newer versions of PyTorch have a stricter default for loading model files for security.
# This patch tells the TTS library's loading function to be less strict,
# ensuring we can correctly load the pre-trained Coqui model checkpoint we downloaded.
original_load_fsspec = io.load_fsspec
def patched_load_fsspec(path, map_location=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load_fsspec(path, map_location, **kwargs)
io.load_fsspec = patched_load_fsspec


def setup_tts_model(model_dir: str) -> Xtts:
    """
    Handles the setup and loading of the local, offline Coqui XTTS-v2 model.
    This is a heavyweight, time-consuming operation, so our script is designed
    to call this function only once at the very beginning for efficiency.
    """
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] --- Setting up Coqui TTS Model ---")
    
    config_path = os.path.join(model_dir, "config.json")
    model_file = os.path.join(model_dir, "model.pth")
    vocab_file = os.path.join(model_dir, "vocab.json")

    # Before proceeding, we check that all the essential model files are present.
    for f in [config_path, model_file, vocab_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"FATAL: Required TTS model file not found at {f}")
    
    print(f"[{time.time() - start_time:.2f}s] All model files found. Loading model configuration...")
    config = XttsConfig()
    config.load_json(config_path)
    print(f"[{time.time() - start_time:.2f}s] ✓ Configuration loaded.")
    
    print(f"[{time.time() - start_time:.2f}s] Initializing model from configuration...")
    model = Xtts.init_from_config(config)
    print(f"[{time.time() - start_time:.2f}s] ✓ Model initialized.")
    
    print(f"[{time.time() - start_time:.2f}s] Loading model weights from checkpoint... (This is often the slowest step)")
    model.load_checkpoint(config, checkpoint_dir=model_dir, use_deepspeed=False)
    print(f"[{time.time() - start_time:.2f}s] ✓ Model weights loaded.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.time() - start_time:.2f}s] Moving TTS model to device: {device}...")
    model.to(device)
    print(f"[{time.time() - start_time:.2f}s] ✓ Model moved to GPU.")
    
    print(f"[{time.time() - start_time:.2f}s] --- TTS Model setup complete and ready for inference. ---")
    return model


def process_dataset(tts_model: Xtts, speaker_wav: str, experiment_name: str, dataset_name: str, results_dir: str, output_audio_root: str):
    """
    This is the main workhorse function. It processes a single dataset for a
    given experiment, converting all its reasoning chains to audio files.
    """
    print(f"\n--- Processing Dataset: {dataset_name.upper()} for Experiment: {experiment_name.upper()} ---")

    # We construct the path to the input file based on the 'default' condition results.
    input_jsonl_path = os.path.join(results_dir, experiment_name, f"{experiment_name}_{dataset_name}.jsonl")
    if not os.path.exists(input_jsonl_path):
        print(f"  - WARNING: Results file not found, skipping: {input_jsonl_path}")
        return

    # We create the structured output directory for the new audio files.
    output_dir = os.path.join(output_audio_root, experiment_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  - Reading trials from: {input_jsonl_path}")
    all_trials = [json.loads(line) for line in open(input_jsonl_path, 'r', encoding='utf-8')]
    print(f"  - Found {len(all_trials)} trials to process.")
    print(f"  - Saving generated audio to: {output_dir}")

    files_generated_this_run = 0
    for i, trial in enumerate(all_trials):
        try:
            # --- Step 1: Extract the correct reasoning text ---
            if experiment_name == 'baseline':
                text_to_speak = trial.get('sanitized_cot', '')
            else:
                text_to_speak = trial.get('final_prompt_messages', [{}, {'content': ''}])[1]['content']

            if not text_to_speak.strip():
                continue

            # --- Step 2: Construct the unique, predictable output filename ---
            base_name = f"{experiment_name}_{trial['id']}_{trial['chain_id']}"
            if experiment_name == 'early_answering':
                filename = f"{base_name}_step_{trial['num_sentences_provided']}.wav"
            elif experiment_name == 'paraphrasing':
                filename = f"{base_name}_paraphrased_{trial['num_sentences_paraphrased']}.wav"
            elif experiment_name == 'adding_mistakes':
                filename = f"{base_name}_mistake_{trial['mistake_position']}.wav"
            else:
                filename = f"{base_name}.wav"
            
            output_wav_path = os.path.join(output_dir, filename)

            # --- Step 3: Check for existence (The HPC Restartability Feature) ---
            if os.path.exists(output_wav_path):
                continue
            
            if files_generated_this_run % 20 == 0:
                 print(f"  - [Trial {i+1}/{len(all_trials)}] Generating: {filename}")

            # --- Step 4: Run TTS Inference ---
            outputs = tts_model.synthesize(
                text_to_speak, tts_model.config, speaker_wav=speaker_wav,
                language="en", enable_text_splitting=True
            )
            
            # The raw output from the model is a NumPy array.
            numpy_audio = outputs['wav']
            
            # --- THE CRITICAL FIX ---
            # We must explicitly convert the NumPy array to a PyTorch tensor
            # before we can perform tensor operations (like .dim() or .unsqueeze()) on it.
            audio_tensor = torch.from_numpy(numpy_audio).float()
            # --- END OF FIX ---
            
            # Now that it's a tensor, we can safely check its dimensions.
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # --- Step 5: Save the Audio File ---
            torchaudio.save(output_wav_path, audio_tensor.cpu(), sample_rate=24000)
            files_generated_this_run += 1

        except Exception as e:
            print(f"  - ✗ ERROR on trial {i+1} (ID: {trial.get('id', 'N/A')}): {e}. Skipping.")
            continue


if __name__ == "__main__":
    # This is a standard solution for a common problem on HPC clusters where
    # libraries that use multiprocessing (like TTS) can hang when interacting with CUDA.
    # By setting the start method to 'spawn', we ensure that any new background
    # processes are created in a clean, safe way, avoiding deadlocks.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # This might raise an error if the context is already set, which is fine.
        pass
    
    parser = argparse.ArgumentParser(
        description="Generate spoken reasoning audio files from experiment results using Coqui TTS.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--experiment', type=str, required=True, choices=['baseline', 'early_answering', 'paraphrasing', 'adding_mistakes'], help="The name of the experiment whose results you want to convert to audio.")
    parser.add_argument('--dataset', type=str, required=True, help="The short name of the dataset to process (e.g., 'mmar' or 'all').")
    parser.add_argument('--results_dir', type=str, default='./results', help="Root directory of the 'default' condition experiment results.")
    parser.add_argument('--output_dir', type=str, default='./spoken_reasoning/audio', help="Root directory to save the generated audio files.")
    parser.add_argument('--tts_model_dir', type=str, default='./tts_models/XTTS-v2', help="Path to the local XTTS-v2 model directory.")
    parser.add_argument('--speaker_wav', type=str, default='./tts_models/reference_speaker.wav', help="Path to the reference speaker audio file.")
    args = parser.parse_args()

    # --- Setup ---
    tts_model = setup_tts_model(args.tts_model_dir)
    
    # --- Main Logic ---
    start_time = time.time()
    
    if args.dataset == 'all':
        experiment_results_dir = os.path.join(args.results_dir, args.experiment)
        if not os.path.exists(experiment_results_dir):
            print(f"FATAL: No results directory found for experiment '{args.experiment}' at '{experiment_results_dir}'")
            exit(1)
        
        result_files = [f for f in os.listdir(experiment_results_dir) if f.endswith('.jsonl') and not f.endswith(('_default.jsonl', '_transcribed_audio.jsonl'))]
        dataset_names = sorted([f.replace(f"{args.experiment}_", "").replace(".jsonl", "") for f in result_files])
        
        print(f"\nFound {len(dataset_names)} datasets to process for the '{args.experiment}' experiment: {dataset_names}")
        for dataset in dataset_names:
            process_dataset(tts_model, args.speaker_wav, args.experiment, dataset, args.results_dir, args.output_dir)
    else:
        process_dataset(tts_model, args.speaker_wav, args.experiment, args.dataset, args.results_dir, args.output_dir)
    
    end_time = time.time()
    print(f"\n--- Full generation process completed in {end_time - start_time:.2f} seconds. ---")