# data_processing/generate_spoken_reasoning.py

import os
import json
import argparse
import torch
import torchaudio
import time
import multiprocessing as mp
import numpy as np
import math

# We add pydub for audio segment manipulation (stitching clips together).
try:
    from pydub import AudioSegment
except ImportError:
    print("FATAL ERROR: The 'pydub' library was not found.")
    print("Please install it in your 'tts-env' Conda environment with: pip install pydub")
    exit(1)

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils import io
except ImportError:
    print("FATAL ERROR: The 'TTS' library was not found.")
    print("Please ensure you have activated the dedicated 'tts-env' Conda environment.")
    exit(1)

# --- Monkey Patch and Model Setup (Unchanged) ---
original_load_fsspec = io.load_fsspec
def patched_load_fsspec(path, map_location=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load_fsspec(path, map_location, **kwargs)
io.load_fsspec = patched_load_fsspec

def setup_tts_model(model_dir: str) -> Xtts:
    # This function remains unchanged.
    # ... (same as your provided script)
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] --- Setting up Coqui TTS Model ---")
    config_path = os.path.join(model_dir, "config.json")
    model_file = os.path.join(model_dir, "model.pth")
    vocab_file = os.path.join(model_dir, "vocab.json")
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

# --- NEW HELPER FUNCTION FOR CONCATENATION ---
def generate_concatenated_audio(tts_model: Xtts, text: str, speaker_wav: str, chunk_size: int = 350) -> torch.Tensor:
    """
    Generates audio for long texts by splitting them into chunks, generating
    audio for each chunk, and concatenating the results.
    """
    # We use the model's own tokenizer to split the text into tokens.
    tokens = tts_model.tokenizer.encode(text, lang= 'en')
    
    # We then split these tokens into safe-sized chunks.
    token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    audio_segments = []
    for i, chunk in enumerate(token_chunks):
        # Convert the tokens for this chunk back to text.
        chunk_text = tts_model.tokenizer.decode(chunk)
        
        # Generate audio for this smaller piece of text.
        outputs = tts_model.synthesize(
            chunk_text, tts_model.config, speaker_wav=speaker_wav,
            language="en", enable_text_splitting=True # Keep splitting enabled for safety
        )
        
        # The output is a numpy array, which we convert to a pydub AudioSegment.
        numpy_audio = outputs['wav']
        # Convert numpy array to raw audio data that pydub can understand
        raw_audio_data = (numpy_audio * 32767).astype(np.int16).tobytes()
        segment = AudioSegment(
            data=raw_audio_data,
            sample_width=2, # 16-bit audio
            frame_rate=24000,
            channels=1
        )
        audio_segments.append(segment)

    # Stitch all the generated audio segments together.
    concatenated_audio = sum(audio_segments)

    # Convert the final pydub segment back to a PyTorch tensor for saving.
    final_numpy_audio = np.array(concatenated_audio.get_array_of_samples()).astype(np.float32) / 32767.0
    return torch.from_numpy(final_numpy_audio)


def process_dataset(tts_model: Xtts, speaker_wav: str, experiment_name: str, dataset_name: str, results_dir: str, output_audio_root: str):
    """
    Main workhorse function. This version now handles long CoTs via concatenation.
    """
    # ... (Setup and file loading is unchanged) ...
    print(f"\n--- Processing Dataset: {dataset_name.upper()} for Experiment: {experiment_name.upper()} ---")
    input_jsonl_path = os.path.join(results_dir, experiment_name, f"{experiment_name}_{dataset_name}.jsonl")
    if not os.path.exists(input_jsonl_path):
        print(f"  - WARNING: Results file not found, skipping: {input_jsonl_path}")
        return
    output_dir = os.path.join(output_audio_root, experiment_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  - Reading trials from: {input_jsonl_path}")
    all_trials = [json.loads(line) for line in open(input_jsonl_path, 'r', encoding='utf-8')]
    print(f"  - Found {len(all_trials)} trials to process.")
    print(f"  - Saving generated audio to: {output_dir}")
    
    failed_trials = []
    files_generated_this_run = 0
    for i, trial in enumerate(all_trials):
        try:
            # ... (Text extraction and filename construction are unchanged) ...
            if experiment_name == 'baseline':
                text_to_speak = trial.get('sanitized_cot', '')
            else:
                text_to_speak = trial.get('final_prompt_messages', [{}, {'content': ''}])[1]['content']
            if not text_to_speak.strip():
                continue
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

            if os.path.exists(output_wav_path):
                continue
            
            if files_generated_this_run % 20 == 0:
                 print(f"  - [Trial {i+1}/{len(all_trials)}] Generating: {filename}")

            # --- THE NEW LOGIC: Check token length and decide which method to use ---
            tokens = tts_model.tokenizer.encode(text_to_speak, lang = 'en')
            if len(tokens) > 380:
                # If the text is too long, use our new concatenation helper.
                print(f"    - INFO: Text exceeds 380 tokens ({len(tokens)}). Using concatenation.")
                audio_tensor = generate_concatenated_audio(tts_model, text_to_speak, speaker_wav)
            else:
                # Otherwise, use the standard, direct synthesis method.
                outputs = tts_model.synthesize(
                    text_to_speak, tts_model.config, speaker_wav=speaker_wav,
                    language="en", enable_text_splitting=True
                )
                numpy_audio = outputs['wav']
                audio_tensor = torch.from_numpy(numpy_audio).float()
            
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            torchaudio.save(output_wav_path, audio_tensor.cpu(), sample_rate=24000)
            files_generated_this_run += 1

        except Exception as e:
            # Our robust failure logging.
            error_message = str(e)
            failed_trials.append({"trial_index": i, "id": trial.get('id', 'N/A'), "error": error_message})
            print(f"  - ✗ ERROR on trial {i+1} (ID: {trial.get('id', 'N/A')}): {error_message}. Skipping.")
            continue
    
    # --- Write Failure Log ---
    if failed_trials:
        log_filename = f"tts_generation_failures_{experiment_name}_{dataset_name}.log"
        log_path = os.path.join(output_audio_root, experiment_name, log_filename)
        with open(log_path, 'w') as f:
            json.dump(failed_trials, f, indent=2)
        print(f"\n[!] WARNING: {len(failed_trials)} trials failed. A detailed log has been saved to: {log_path}")


# --- Main block is unchanged ---
if __name__ == "__main__":
    # ... (same as your provided script)
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
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
    tts_model = setup_tts_model(args.tts_model_dir)
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