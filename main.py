# main.py

import argparse
import sys
import os
import importlib

import nltk

# Add the project root to the Python path for robust imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.lalm_utils import load_model_and_tokenizer
from data_loader.data_loader import load_dataset

# This block runs once when the program starts, ensuring NLTK is
# configured correctly for all subsequent experiment scripts.
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
    # Optional: A check to ensure the specific package we need is present.
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print(f"FATAL: NLTK 'punkt' model not found in '{local_nltk_data_path}'.")
        exit(1)
else:
    print(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    exit(1)


# ==============================================================================
#  Dataset Alias Mapping
# ==============================================================================

DATASET_MAPPING = {
    # --- Original Clean Datasets ---
    "mmar": "data/mmar/mmar_test_standardized.jsonl",
    "sakura-animal": "data/sakura/animal/sakura_animal_test_standardized.jsonl",
    "sakura-emotion": "data/sakura/emotion/sakura_emotion_standardized.jsonl",
    "sakura-gender": "data/sakura/gender/sakura_gender_standardized.jsonl",
    "sakura-language": "data/sakura/language/sakura_language_standardized.jsonl",

    # --- NEW: Noisy Datasets ---
    "mmar-noisy": "data/mmar_noisy/mmar_noisy_standardized.jsonl",
    "sakura-animal-noisy": "data/sakura_noisy/animal/animal_noisy_standardized.jsonl",
    "sakura-emotion-noisy": "data/sakura_noisy/emotion/emotion_noisy_standardized.jsonl",
    "sakura-gender-noisy": "data/sakura_noisy/gender/gender_noisy_standardized.jsonl",
    "sakura-language-noisy": "data/sakura_noisy/language/language_noisy_standardized.jsonl",
}

def main():
    parser = argparse.ArgumentParser(
        description="Run LALM Faithfulness Experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--experiment", type=str, required=True, help="The name of the experiment module to run (e.g., 'baseline').")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_MAPPING.keys(), help="The short name of the dataset to use.")
    parser.add_argument("--baseline-results-file", type=str, default=None, help="(For dependent experiments) Path to a specific baseline results file to use as input.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument('--verbose', action='store_true', help="Enable detailed, line-by-line progress logging for long-running experiments.")

    args = parser.parse_args()

    # --- 1. Configuration Setup ---
    if args.num_samples is not None:
        config.NUM_SAMPLES_TO_RUN = args.num_samples
    if args.num_chains:
        config.NUM_CHAINS_PER_QUESTION = args.num_chains
    
    config.DATASET_NAME = args.dataset
    
    config.VERBOSE = args.verbose

    # Centralized Output Path Management
    # The orchestrator is now responsible for defining the output path.
    # This creates the desired structure: results/{experiment_name}/{experiment_name}_{dataset_name}.jsonl
    experiment_name = args.experiment
    output_dir = os.path.join(config.RESULTS_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{experiment_name}_{config.DATASET_NAME}.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    
    # We pass the final path to the experiment via the config object.
    config.OUTPUT_PATH = output_path
    

    print("--- LALM Faithfulness Framework ---")
    print(f"  - Experiment: {args.experiment}")
    print(f"  - Dataset:    {config.DATASET_NAME}")
    print(f"  - Outputting to: {config.OUTPUT_PATH}")
    print(f"  - Verbose Logging: {'Enabled' if config.VERBOSE else 'Disabled'}")
    print("-" * 35)

    # --- 2. Dynamic Experiment Dispatch ---
    try:
        print(f"Loading experiment module: 'experiments.{args.experiment}'...")
        experiment_module = importlib.import_module(f"experiments.{args.experiment}")
        EXPERIMENT_TYPE = getattr(experiment_module, 'EXPERIMENT_TYPE')
        print(f"Detected experiment type: '{EXPERIMENT_TYPE}'")
    except (ImportError, AttributeError):
        print(f"FATAL: Could not load experiment '{args.experiment}'.")
        print(f"Ensure 'experiments/{args.experiment}.py' exists and contains the 'EXPERIMENT_TYPE' variable.")
        sys.exit(1)

    # --- 3. Load Model (Common to all experiments) ---
    print("\nLoading model and processor...")
    model, processor = load_model_and_tokenizer(config.MODEL_PATH)

    # --- 4. Execute based on Experiment Type ---
    if EXPERIMENT_TYPE == "foundational":
        print("\nRunning a FOUNDATIONAL experiment...")
        dataset_path = DATASET_MAPPING[args.dataset]
        try:
            data_samples = load_dataset(dataset_path)
            if config.NUM_SAMPLES_TO_RUN > 0:
                data_samples = data_samples[:config.NUM_SAMPLES_TO_RUN]
            print(f"Processing {len(data_samples)} samples.")
            experiment_module.run(model, processor, data_samples, config)
        except FileNotFoundError:
            print(f"FATAL: Dataset file not found at '{dataset_path}'.")
            sys.exit(1)

    elif EXPERIMENT_TYPE == "dependent":
        print("\nRunning a DEPENDENT experiment...")
        config.BASELINE_RESULTS_FILE_OVERRIDE = args.baseline_results_file
        experiment_module.run(model, processor, config)

    else:
        print(f"FATAL: Unknown experiment type '{EXPERIMENT_TYPE}' in 'experiments/{args.experiment}.py'.")
        sys.exit(1)

    print("\n--- Experiment completed successfully! ---")

if __name__ == "__main__":
    main()