# main.py

import argparse
import sys
import os
import importlib
import nltk

# To make sure our project's modules can be found, we add the root directory to the Python path.
# This makes our imports cleaner and more reliable, no matter where we run the script from.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Environment Setup: NLTK Data Path ---
# Before we do anything else, we need to make sure our environment is set up correctly.
# This block tells the NLTK library where to find our local, offline 'punkt' package for sentence tokenization.
# We do this here, once, at the very beginning, so that every other script in our project
# can use NLTK without worrying about its configuration. This is a crucial step for
# running our code successfully on the firewalled HPC compute nodes.
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
    try:
        # We also do a quick check to make sure the specific 'punkt' package we need is actually there.
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print(f"FATAL: NLTK 'punkt' model not found in '{local_nltk_data_path}'.")
        exit(1)
else:
    print(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    exit(1)

# Now that the environment is set, we can safely import our own project modules.
import config
from core.lalm_utils import load_model_and_tokenizer
from data_loader.data_loader import load_dataset

def main():
    """
    This is the main entry point and orchestrator for our entire research framework.
    Its job is to parse user commands, set up the global configuration, and then
    delegate the actual scientific work to the appropriate experiment script.
    """
    
    # --- Command-Line Interface (CLI) Setup ---
    # This is our "command center." We use argparse to define all the flags and
    # options a user can provide to control the experiments.
    parser = argparse.ArgumentParser(
        description="Run LALM Faithfulness Experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    dataset_choices = [
        "mmar",
        "sakura-animal",
        "sakura-emotion",
        "sakura-gender",
        "sakura-language"
    ]
    
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_choices, help="The base name of the dataset to use (e.g., 'mmar', 'sakura-animal').")
    parser.add_argument("--experiment", type=str, required=True, help="The name of the experiment module to run (e.g., 'baseline').")
    
    # This is the new, powerful flag that lets us switch between our major experimental conditions.
    parser.add_argument(
        "--condition", 
        type=str, 
        default="default", 
        choices=["default", "transcribed_audio", "spoken_reasoning"],
        help="The experimental condition to run. 'default' uses the original audio."
    )
    
    # These are optional arguments for controlling the scale of the run, perfect for quick tests.
    parser.add_argument("--baseline-results-file", type=str, default=None, help="(For dependent experiments) Path to a specific baseline results file to use as input.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument('--verbose', action='store_true', help="Enable detailed, line-by-line progress logging.")

    args = parser.parse_args()

    # --- 1. Global Configuration Setup ---
    # Here, we take the user's command-line arguments and use them to set global
    # variables in our 'config' module. This is a powerful design because it makes
    # these settings accessible to every other script in the project with a simple 'import config'.
    if args.num_samples is not None: config.NUM_SAMPLES_TO_RUN = args.num_samples
    if args.num_chains: config.NUM_CHAINS_PER_QUESTION = args.num_chains
    
    config.DATASET_NAME = args.dataset
    config.VERBOSE = args.verbose
    config.CONDITION = args.condition # This makes the current condition globally known.

    # --- 2. Centralized Path Management ---
    # To keep our experiment scripts clean, the orchestrator is responsible for figuring out
    # exactly where the results should be saved.
    experiment_name = args.experiment
    
    # We create a clear, descriptive filename that includes the experiment, dataset, and condition.
    # This makes our results folders easy to navigate and understand.
    output_filename = f"{experiment_name}_{args.dataset}_{args.condition}.jsonl"
    
    output_dir = os.path.join(config.RESULTS_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # We save the final, complete path to the config object, so the experiment script
    # doesn't need to think about it at all. It just saves to 'config.OUTPUT_PATH'.
    config.OUTPUT_PATH = os.path.join(output_dir, output_filename)
    
    # This is the logic that intelligently finds the right data file to load based on the user's command.
    if args.condition == 'default':
        dataset_key = args.dataset
    else:
        # This creates the key we need to look up in our config file's DATASET_MAPPING.
        # e.g., 'sakura-animal' + 'transcribed_audio' -> 'sakura-animal-transcribed_audio'
        dataset_key = f"{args.dataset}-{args.condition}"

    # --- 3. Print a "Run Summary" Banner ---
    # This is helpful for confirming that the job is starting with the correct settings.
    print("--- LALM Faithfulness Framework ---")
    print(f"  - Experiment: {args.experiment}")
    print(f"  - Dataset:    {args.dataset}")
    print(f"  - Condition:  {config.CONDITION}")
    print(f"  - Outputting to: {config.OUTPUT_PATH}")
    print(f"  - Verbose Logging: {'Enabled' if config.VERBOSE else 'Disabled'}")
    print("-" * 35)

    # --- 4. Dynamic Experiment Loading ---
    # This is the "magic" that makes our framework so flexible. Instead of a giant
    # if/else block for every experiment, we dynamically import the experiment script
    # the user asked for.
    try:
        experiment_module = importlib.import_module(f"experiments.{args.experiment}")
        # We then check for the 'EXPERIMENT_TYPE' variable inside that script. This is like a
        # "contract" that tells the orchestrator how to treat the experiment.
        EXPERIMENT_TYPE = getattr(experiment_module, 'EXPERIMENT_TYPE')
    except (ImportError, AttributeError):
        print(f"FATAL: Could not load experiment '{args.experiment}'.")
        print(f"Ensure 'experiments/{args.experiment}.py' exists and contains the 'EXPERIMENT_TYPE' variable.")
        sys.exit(1)

    print(f"Detected experiment type: '{EXPERIMENT_TYPE}'")
    
    # --- 5. Load the Model ---
    # Since all experiments use the same model, we load it once here.
    print("\nLoading model and processor...")
    model, processor = load_model_and_tokenizer(config.MODEL_PATH)

    # --- 6. Execute the Experiment Based on its Type ---
    # This is where the 'EXPERIMENT_TYPE' contract is enforced.
    if EXPERIMENT_TYPE == "foundational":
        # Foundational experiments (like 'baseline') run on the raw, standardized datasets.
        print("\nRunning a FOUNDATIONAL experiment...")
        try:
            dataset_path = config.DATASET_MAPPING[dataset_key]
            data_samples = load_dataset(dataset_path)
            
            # Apply the --num-samples limit if the user provided one.
            if config.NUM_SAMPLES_TO_RUN > 0:
                data_samples = data_samples[:config.NUM_SAMPLES_TO_RUN]
                
            print(f"Processing {len(data_samples)} samples from '{dataset_path}'.")
            # We pass the loaded data directly to the experiment's 'run' function.
            experiment_module.run(model, processor, data_samples, config)
        except KeyError:
            print(f"FATAL: Dataset key '{dataset_key}' not found in DATASET_MAPPING in config.py.")
            sys.exit(1)
        except FileNotFoundError:
            print(f"FATAL: Dataset file not found for key '{dataset_key}'. Check the path in config.py.")
            sys.exit(1)

    elif EXPERIMENT_TYPE == "dependent":
        # Dependent experiments (like 'filler_text') need the results from a baseline run.
        # We don't pass them data directly. Instead, they are responsible for loading their
        # own data from the baseline results files.
        print("\nRunning a DEPENDENT experiment...")
        config.BASELINE_RESULTS_FILE_OVERRIDE = args.baseline_results_file
        experiment_module.run(model, processor, config)

    else:
        # This is a safety check to ensure the 'EXPERIMENT_TYPE' is a known value.
        print(f"FATAL: Unknown experiment type '{EXPERIMENT_TYPE}' in 'experiments/{args.experiment}.py'.")
        sys.exit(1)

    print("\n--- Experiment completed successfully! ---")

if __name__ == "__main__":
    main()