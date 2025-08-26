# data_processing/generate_spoken_reasoning.py

import os
import json
import argparse
from pathlib import Path
import torch
from TTS.api import TTS
import sys

# Add the project root to the Python path to allow importing our core modules
# This is necessary if we ever want to import something from `core` or `config`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
# Define the local paths to our TTS assets. These must be correct.
TTS_MODEL_PATH = "tts_models/XTTS-v2/"
REFERENCE_SPEAKER_PATH = "tts_models/reference_speaker.wav"
DEFAULT_RESULTS_DIR = "results/"
DEFAULT_OUTPUT_DIR = "spoken_reasoning/audio/"

def get_reasoning_text(trial: dict) -> str:
    """
    A helper function to reliably extract the assistant's reasoning text
    from a trial's 'final_prompt_messages' list. This text is what we will convert to speech.
    """
    # We look for the 'assistant' role, as that contains the CoT.
    if 'final_prompt_messages' not in trial:
        return ""
    for message in trial['final_prompt_messages']:
        if message.get('role') == 'assistant':
            return message.get('content', '')
    return ""

def construct_output_filename(experiment_name: str, trial: dict) -> str:
    """
    Creates a unique, predictable, and descriptive filename for a given trial's audio.
    This is the core of our strategy to avoid creating new intermediate JSONL files.
    The filename itself becomes our metadata lookup key.
    """
    q_id = trial['id']
    chain_id = trial['chain_id']
    
    # This 'router' correctly handles the unique key for each experiment type.
    if experiment_name == 'early_answering':
        step = trial['num_sentences_provided']
        return f"early_answering_{q_id}_chain_{chain_id}_step_{step}.wav"
    elif experiment_name == 'paraphrasing':
        step = trial['num_sentences_paraphrased']
        return f"paraphrasing_{q_id}_chain_{chain_id}_step_{step}.wav"
    elif experiment_name == 'adding_mistakes':
        step = trial['mistake_position']
        return f"adding_mistakes_{q_id}_chain_{chain_id}_step_{step}.wav"
    elif experiment_name == 'baseline':
        # The baseline has no 'step', so its filename is simpler.
        return f"baseline_{q_id}_chain_{chain_id}.wav"
    else:
        # This provides a safe fallback, though we don't expect to use it.
        print(f"WARNING: Unhandled experiment type for filename construction: {experiment_name}")
        return f"{experiment_name}_{q_id}_chain_{chain_id}.wav"

def process_dataset(tts_model, experiment_name: str, dataset_name: str, results_dir: str, output_dir: str):
    """
    Processes a single results file (e.g., 'early_answering_mmar_default.jsonl'),
    generating a spoken reasoning audio file for every trial it contains.
    """
    # We are always generating spoken reasoning from the 'default' (original audio) condition results.
    results_filename = f"{experiment_name}_{dataset_name}.jsonl"
    results_filepath = Path(results_dir) / 'default_experiments' / experiment_name / results_filename
    
    if not results_filepath.exists():
        print(f"  - WARNING: Results file not found, skipping: {results_filepath}")
        return

    print(f"\nProcessing {experiment_name} for dataset {dataset_name}...")
    
    # Create the specific, organized output directory for this experiment and dataset's audio.
    specific_output_dir = Path(output_dir) / experiment_name / dataset_name
    specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    trials = [json.loads(line) for line in open(results_filepath, 'r')]
    
    generated_count = 0
    skipped_count = 0
    failed_count = 0

    for i, trial in enumerate(trials):
        # A simple progress indicator for long jobs.
        print(f"  - Processing trial {i+1}/{len(trials)}...", end='\r')
        
        reasoning_text = get_reasoning_text(trial)
        
        # If there's no reasoning text, there's no audio to generate.
        if not reasoning_text.strip():
            continue

        output_filename = construct_output_filename(experiment_name, trial)
        output_filepath = specific_output_dir / output_filename

        # This is the critical "restartable" feature for long HPC jobs.
        if output_filepath.exists():
            skipped_count += 1
            continue

        try:
            # The core TTS generation call using our consistent reference speaker.
            tts_model.tts_to_file(
                text=reasoning_text,
                speaker_wav=REFERENCE_SPEAKER_PATH,
                language="en",
                file_path=str(output_filepath)
            )
            generated_count += 1
        except Exception as e:
            # Robust error handling to prevent one bad trial from killing the whole job.
            print(f"\n  - ERROR generating TTS for trial {trial.get('id', 'N/A')}: {e}")
            failed_count += 1
            continue
            
    print(f"\n  - Done. Generated: {generated_count}, Skipped (already exist): {skipped_count}, Failed: {failed_count}")


if __name__ == "__main__":
    # We define the list of experiments that we are processing in this run.
    # Filler text variants are excluded for now, as per our discussion.
    VALID_EXPERIMENTS = [
        'baseline', 'early_answering', 'paraphrasing', 'adding_mistakes'
    ]

    parser = argparse.ArgumentParser(description="Generate spoken reasoning audio files from experiment results.")
    parser.add_argument('--experiment', type=str, required=True, choices=VALID_EXPERIMENTS + ['all'], help="The experiment results to process.")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset to process (e.g., 'mmar', or 'all').")
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # --- Load the TTS Model Once ---
    # This is a heavyweight operation, so we do it once at the very start.
    print("Initializing Coqui TTS model from local files... (This may take a moment)")
    
    # Instead of the high-level constructor, we use a more explicit method
    # that is designed for loading from a local, offline directory.
    # We point it directly to the model's configuration file.
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import Synthesizer

    # This is the new, robust way to initialize the model for offline use.
    synthesizer = Synthesizer(
        tts_checkpoint=os.path.join(TTS_MODEL_PATH, "model.pth"),
        tts_config_path=os.path.join(TTS_MODEL_PATH, "config.json"),
        use_cuda=torch.cuda.is_available(),
    )

    print("TTS model loaded successfully.")

    experiments_to_process = VALID_EXPERIMENTS if args.experiment == 'all' else [args.experiment]
    
    if args.dataset == 'all':
        # Find all available dataset names from the baseline results directory.
        baseline_dir = Path(args.results_dir) / 'default_experiments' / 'baseline'
        dataset_names = sorted(list(set([f.replace('baseline_', '').replace('.jsonl', '') for f in os.listdir(baseline_dir) if f.endswith('.jsonl')])))
    else:
        dataset_names = [args.dataset]

    print(f"\nFound datasets to process: {dataset_names}")
    print(f"Found experiments to process: {experiments_to_process}")

    for exp_name in experiments_to_process:
        for dataset_name in dataset_names:
            process_dataset(synthesizer, exp_name, dataset_name, args.results_dir, args.output_dir)

    print("\nSpoken reasoning generation complete.")