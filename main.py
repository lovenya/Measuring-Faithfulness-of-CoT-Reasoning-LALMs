# main.py

import argparse
import sys
import os
import importlib
import nltk
from core.logging_utils import setup_logger
import logging

# faulthandler is a great tool for debugging low-level crashes.
import faulthandler
faulthandler.enable()

# To make sure our project's modules can be found, we add the root directory to the Python path.
# This makes our imports cleaner and more reliable, no matter where we run the script from.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Environment Setup: NLTK Data Path ---
# Before we do anything else, we ensure the environment is set up correctly. This block
# tells the NLTK library where to find our local, offline 'punkt' package. We do this
# here, once, at the very beginning, so that every other script can use NLTK without
# worrying about its configuration. This is crucial for running on firewalled HPC nodes.
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info(f"FATAL: NLTK 'punkt' model not found in '{local_nltk_data_path}'.")
        exit(1)
else:
    logging.info(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    exit(1)

# Now that the environment is set, we can safely import our own project modules.
import config
from data_loader.data_loader import load_dataset

def main():
    """
    This is the main entry point and orchestrator for our research framework.
    Its job is to parse user commands, set up the configuration, dynamically load the
    correct model and experiment, and then delegate the scientific work.
    """
    
    # --- Command-Line Interface (CLI) Setup ---
    # This is our "command center." We define all the flags a user can provide.
    parser = argparse.ArgumentParser(
        description="Run LALM Faithfulness Experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # The --model flag is the primary way to switch between different models.
    # The choices are read dynamically from our config file, which is robust.
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=config.MODEL_ALIASES.keys(),
        help="The alias of the model to use for the experiment (e.g., 'qwen', 'flamingo', 'salmonn')."
    )
    
    # The --dataset choices are also read dynamically from the config file.
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=config.DATASET_MAPPING.keys(), 
        help="The base name of the dataset to use (e.g., 'mmar', 'sakura-animal')."
    )
    
    parser.add_argument("--experiment", type=str, required=True, help="The name of the experiment module to run (e.g., 'baseline').")
    
    # --- NEW: The --restricted Flag ---
    # This is our new, powerful flag for controlling which subset of the data to run on.
    # It's a boolean flag, meaning you just add '--restricted' to the command to activate it.
    parser.add_argument(
        '--restricted', 
        action='store_true', 
        help="If specified, run the experiment on the '-restricted' subset of data (3, 4, 5, 6-step CoTs)."
    )
    
    # Optional arguments for controlling the scale of the run, perfect for quick tests.
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument('--verbose', action='store_true', help="Enable detailed, line-by-line progress logging.")

    args = parser.parse_args()

    # --- 1. Global Configuration Setup ---
    # We take the user's commands and set global variables in our 'config' module.
    # This makes these settings easily accessible to every other script in the project.
    if args.num_samples is not None: config.NUM_SAMPLES_TO_RUN = args.num_samples
    if args.num_chains: config.NUM_CHAINS_PER_QUESTION = args.num_chains
    
    config.MODEL_ALIAS = args.model
    config.DATASET_NAME = args.dataset
    config.VERBOSE = args.verbose
    # We make the state of the --restricted flag globally available to all scripts.
    config.RESTRICTED = args.restricted

    # --- 2. Centralized Path Management ---
    # This block intelligently constructs the output path to keep our results organized
    # by model, preventing different models from overwriting each other's data.
    experiment_name = args.experiment
    model_alias = config.MODEL_ALIAS
    
    # e.g., 'results/qwen/baseline/' or 'results/salmonn/filler_text/'
    output_dir = os.path.join(config.RESULTS_DIR, model_alias, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # The output filename now also reflects if it was a restricted run.
    # e.g., 'adding_mistakes_salmonn_mmar-restricted.jsonl'
    if config.RESTRICTED:
        output_filename = f"{experiment_name}_{model_alias}_{args.dataset}-restricted.jsonl"
    else:
        output_filename = f"{experiment_name}_{model_alias}_{args.dataset}.jsonl"
    config.OUTPUT_PATH = os.path.join(output_dir, output_filename)
    
    log_dir = os.path.join(config.RESULTS_DIR, 'logs')
    log_filepath = setup_logger(log_dir, model_alias, experiment_name, args.dataset)
    
    # --- 3. Dynamic Model Utility Loading ---
    # This is the core of our multi-model architecture. Based on the --model flag,
    # we dynamically import the correct utility file (e.g., 'qwen_utils.py').
    # We give it a standard alias, 'model_utils', so all subsequent code can call
    # functions like 'model_utils.load_model_and_tokenizer()' without needing to
    # know which specific model is being used.
    logging.info(f"Loading utility module for model: {model_alias}")
    try:
        model_key = config.MODEL_ALIASES[model_alias]
        model_path = config.MODEL_PATHS[model_key]
        
        if model_alias == 'qwen':
            from core import qwen_utils as model_utils
        elif model_alias == 'flamingo':
            from core import audio_flamingo_utils as model_utils
        elif model_alias == 'salmonn':
            from core import salmonn_utils as model_utils
        # To add a new model, you would add a new 'elif' block here.
        else:
            raise ImportError(f"No utility module defined for model alias '{model_alias}'")

    except (ImportError, KeyError) as e:
        logging.exception(f"Could not load utilities for model '{model_alias}'. Check config and core directory.")
        sys.exit(1)

    # --- 4. logging.info a "Run Summary" Banner ---
    logging.info("--- LALM Faithfulness Framework ---")
    logging.info(f"  - Model:      {model_alias.upper()}")
    logging.info(f"  - Experiment: {args.experiment}")
    logging.info(f"  - Dataset:    {args.dataset}")
    # We now clearly log whether this is a restricted run.
    logging.info(f"  - Run Mode:   {'RESTRICTED' if config.RESTRICTED else 'FULL DATASET'}")
    logging.info(f"  - Outputting to: {config.OUTPUT_PATH}")
    logging.info(f"  - Verbose Logging: {'Enabled' if config.VERBOSE else 'Disabled'}")
    logging.info(f"  - Full log will be saved to: {log_filepath}")
    logging.info("-" * 35)

    # --- 5. Dynamic Experiment Loading ---
    # This logic dynamically imports the requested experiment script.
    try:
        experiment_module = importlib.import_module(f"experiments.{args.experiment}")
        EXPERIMENT_TYPE = getattr(experiment_module, 'EXPERIMENT_TYPE')
    except (ImportError, AttributeError):
        logging.exception(f"Could not load experiment '{args.experiment}'.")
        sys.exit(1)

    logging.info(f"Detected experiment type: '{EXPERIMENT_TYPE}'")
    
    # --- 6. Load the Model using the Dynamic Utilities ---
    logging.info("Loading model and processor...")
    # This call now works for any model because of our 'model_utils' alias.
    model, processor = model_utils.load_model_and_tokenizer(model_path)

    # --- 7. Execute the Experiment Based on its Type ---
    if EXPERIMENT_TYPE == "foundational":
        logging.info("Running a FOUNDATIONAL experiment...")
        try:
            dataset_path = config.DATASET_MAPPING[args.dataset]
            data_samples = load_dataset(dataset_path)
            
            if config.NUM_SAMPLES_TO_RUN > 0:
                data_samples = data_samples[:config.NUM_SAMPLES_TO_RUN]
                
            logging.info(f"Processing {len(data_samples)} samples from '{dataset_path}'.")
            # We pass the dynamically loaded model_utils to the experiment's run function.
            experiment_module.run(model, processor, model_utils, data_samples, config)
        except (KeyError, FileNotFoundError):
            logging.exception("Could not load dataset.")
            sys.exit(1)

    elif EXPERIMENT_TYPE == "dependent":
        logging.info("Running a DEPENDENT experiment...")
        
        # --- NEW: Centralized Logic for Finding Dependency Files ---
        # This is the new, intelligent block that finds the correct baseline file
        # based on whether the run is restricted or not.
        baseline_dir = os.path.join(config.RESULTS_DIR, model_alias, 'baseline')
        if config.RESTRICTED:
            baseline_filename = f"baseline_{model_alias}_{args.dataset}-restricted.jsonl"
        else:
            baseline_filename = f"baseline_{model_alias}_{args.dataset}.jsonl"
        
        # We save the final, correct path to the global config.
        # This is the "Single Source of Truth" for all dependent scripts.
        config.BASELINE_RESULTS_PATH = os.path.join(baseline_dir, baseline_filename)
        
        # We do the same for the no_reasoning file.
        no_reasoning_dir = os.path.join(config.RESULTS_DIR, model_alias, 'no_reasoning')
        if config.RESTRICTED:
            no_reasoning_filename = f"no_reasoning_{model_alias}_{args.dataset}-restricted.jsonl"
        else:
            no_reasoning_filename = f"no_reasoning_{model_alias}_{args.dataset}.jsonl"
        config.NO_REASONING_RESULTS_PATH = os.path.join(no_reasoning_dir, no_reasoning_filename)
        
        # Now we can call the experiment's run function.
        experiment_module.run(model, processor, model_utils, config)

    else:
        logging.info(f"FATAL: Unknown experiment type '{EXPERIMENT_TYPE}' in 'experiments/{args.experiment}.py'.")
        sys.exit(1)

    logging.info("--- Experiment completed successfully! ---")

if __name__ == "__main__":
    main()