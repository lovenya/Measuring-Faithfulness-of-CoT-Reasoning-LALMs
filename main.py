# main.py

import argparse
import sys
import os
import importlib

# Add the project root to the Python path for robust imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.lalm_utils import load_model_and_tokenizer
from data_loader.data_loader import load_dataset

# ==============================================================================
#  Dataset Alias Mapping
# ==============================================================================
# This dictionary maps short, user-friendly names to the actual data files.
# To add a new dataset, just add a new entry here.
DATASET_MAPPING = {
    "mmar": "data/mmar/mmar_test_standardized.jsonl",
    "sakura-animal": "data/sakura/animal/sakura_animal_test_standardized.jsonl",
    "sakura-emotion": "data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
    "sakura-gender": "data/sakura/gender/sakura_gender_test_standardized.jsonl",
    "sakura-language": "data/sakura/language/sakura_language_test_standardized.jsonl",
}

def main():
    parser = argparse.ArgumentParser(
        description="Run LALM Faithfulness Experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="The name of the experiment module to run (e.g., 'baseline')."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_MAPPING.keys(), # Choices are the short names
        help="The short name of the dataset to use."
    )
    # This new argument is for dependent experiments
    parser.add_argument(
        "--baseline-results-file",
        type=str,
        default=None,
        help="(For dependent experiments) Path to a specific baseline results file to use as input.\nIf not provided, a default path will be constructed based on the dataset name."
    )
    # Add other config overrides as before
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)

    args = parser.parse_args()

    # --- 1. Configuration Setup ---
    if args.num_samples is not None:
        config.NUM_SAMPLES_TO_RUN = args.num_samples
    if args.num_chains:
        config.NUM_CHAINS_PER_QUESTION = args.num_chains
    
    # Use the alias for the output file name
    config.DATASET_NAME = args.dataset

    print("--- LALM Faithfulness Framework ---")
    print(f"  - Experiment: {args.experiment}")
    print(f"  - Dataset:    {config.DATASET_NAME}")
    # ... other printouts ...
    print("-" * 35)

    # --- 2. Dynamic Experiment Dispatch ---
    try:
        print(f"Loading experiment module: 'experiments.{args.experiment}'...")
        experiment_module = importlib.import_module(f"experiments.{args.experiment}")
        
        # Check the experiment's self-declared type
        EXPERIMENT_TYPE = getattr(experiment_module, 'EXPERIMENT_TYPE', 'unknown')
        print(f"Detected experiment type: '{EXPERIMENT_TYPE}'")

    except ImportError:
        print(f"FATAL: Experiment '{args.experiment}' not found. Ensure 'experiments/{args.experiment}.py' exists.")
        sys.exit(1)
    except AttributeError:
        print(f"FATAL: Experiment module '{args.experiment}' is missing the required 'EXPERIMENT_TYPE' variable.")
        sys.exit(1)

    # --- 3. Load Model (Common to all experiments) ---
    print("\nLoading model and processor...")
    model, processor = load_model_and_tokenizer(config.MODEL_PATH)

    # --- 4. Execute based on Experiment Type ---
    if EXPERIMENT_TYPE == "foundational":
        # These experiments run on raw data.
        print("\nRunning a FOUNDATIONAL experiment...")
        dataset_path = DATASET_MAPPING[args.dataset]
        print(f"Loading raw data from: {dataset_path}")
        try:
            data_samples = load_dataset(dataset_path)
            if config.NUM_SAMPLES_TO_RUN > 0:
                data_samples = data_samples[:config.NUM_SAMPLES_TO_RUN]
            print(f"Processing {len(data_samples)} samples.")
            
            # Call the experiment's run function with the raw data
            experiment_module.run(model, processor, data_samples, config)

        except FileNotFoundError:
            print(f"FATAL: Dataset file not found at '{dataset_path}'.")
            sys.exit(1)

    elif EXPERIMENT_TYPE == "dependent":
        # These experiments depend on baseline results.
        print("\nRunning an DEPENDENT experiment...")
        
        # We must pass the dataset name to the experiment via the config
        # so it knows which baseline file to load by default.
        config.DATASET_NAME = args.dataset
        config.BASELINE_RESULTS_FILE_OVERRIDE = args.baseline_results_file

        # Call the experiment's run function. Note it has a different signature.
        # It is responsible for loading its own data from the baseline file.
        experiment_module.run(model, processor, config)

    else:
        print(f"FATAL: Unknown experiment type '{EXPERIMENT_TYPE}' declared in 'experiments/{args.experiment}.py'.")
        print("Type must be either 'foundational' or 'dependent'.")
        sys.exit(1)

    print("\n--- Experiment completed successfully! ---")

if __name__ == "__main__":
    main()