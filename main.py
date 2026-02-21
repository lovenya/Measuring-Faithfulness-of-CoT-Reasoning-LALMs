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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Environment Setup: NLTK Data Path ---
# This block ensures NLTK knows where to find our local, offline 'punkt' package.
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
    parser = argparse.ArgumentParser(
        description="Run LALM Faithfulness Experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--model", type=str, required=True, choices=config.MODEL_ALIASES.keys(), help="The alias of the model to use.")
    parser.add_argument("--dataset", type=str, required=True, choices=config.DATASET_MAPPING.keys(), help="The base name of the dataset to use.")
    parser.add_argument("--experiment", type=str, required=True, help="The name of the experiment module to run.")
    
    parser.add_argument('--restricted', action='store_true', help="Run on the '-restricted' subset of data (3, 4, 5, 6-step CoTs).")
    
    # --- Arguments for Parallelization ---
    # These flags are the core of the parallel processing strategy.
    # They are used by a Slurm job array to tell each job which chunk of the data it is responsible for.
    parser.add_argument('--part', type=int, default=None, help="The part number of the data chunk to process (e.g., 7).")
    parser.add_argument('--total-parts', type=int, default=None, help="The total number of chunks the data was split into (e.g., 10).")
    
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument('--verbose', action='store_true', help="Enable detailed, line-by-line progress logging.")
    
    # --- Arguments for External Perturbations (Mistral) ---
    parser.add_argument(
        '--use-external-perturbations', 
        action='store_true', 
        help="Use pre-generated perturbations from an external model (Mistral) instead of self-perturbation."
    )
    parser.add_argument(
        '--perturbation-file', 
        type=str, 
        default=None,
        help="Path to the JSONL file containing pre-generated perturbations (required if --use-external-perturbations is set)."
    )
    parser.add_argument(
        '--filler-type',
        type=str,
        default='dots',
        choices=['dots', 'lorem'],
        help="Type of filler for filler text experiments: 'dots' uses '...' (default), 'lorem' uses Lorem Ipsum tokens."
    )
    
    # --- Arguments for Audio Masking Experiments ---
    parser.add_argument(
        '--mask-type',
        type=str,
        default='silence',
        choices=['silence', 'noise'],
        help="Type of masking for audio masking experiments: 'silence' or 'noise'."
    )
    parser.add_argument(
        '--mask-mode',
        type=str,
        default='random',
        choices=['random', 'start', 'end', 'scattered'],
        help="Position mode for audio masking: 'random' (single block), 'start', 'end', or 'scattered' (multiple distributed segments)."
    )

    # --- Arguments for Adversarial Audio Experiments ---
    parser.add_argument(
        '--adversarial-aug',
        type=str,
        default=None,
        choices=['concat', 'overlay'],
        help="Augmentation mode for adversarial experiment: 'concat' or 'overlay'."
    )
    parser.add_argument(
        '--adversarial-variant',
        type=str,
        default=None,
        choices=['correct', 'wrong'],
        help="Adversarial variant: 'correct' (correct choice injected) or 'wrong' (wrong choice injected)."
    )

    args = parser.parse_args()

    # --- 1. Global Configuration Setup ---
    if args.num_samples is not None: config.NUM_SAMPLES_TO_RUN = args.num_samples
    if args.num_chains is not None: config.NUM_CHAINS_PER_QUESTION = args.num_chains
    
    config.MODEL_ALIAS = args.model
    config.DATASET_NAME = args.dataset
    config.VERBOSE = args.verbose
    config.RESTRICTED = args.restricted
    
    # External perturbation settings (for adding_mistakes and paraphrasing experiments)
    config.USE_EXTERNAL_PERTURBATIONS = args.use_external_perturbations
    config.PERTURBATION_FILE = args.perturbation_file
    
    # If running in parallel with --part, update the perturbation file path to use the part file
    # e.g., file_combined.jsonl -> file_combined.part_N.jsonl
    if args.part is not None and config.PERTURBATION_FILE:
        base_path, ext = os.path.splitext(config.PERTURBATION_FILE)
        config.PERTURBATION_FILE = f"{base_path}.part_{args.part}{ext}"
        logging.info(f"[PARALLEL] Using part file: {config.PERTURBATION_FILE}")
    
    # Filler type setting (for filler text experiments)
    config.FILLER_TYPE = args.filler_type
    
    # Audio masking settings (for audio_masking experiment)
    config.MASK_TYPE = args.mask_type
    config.MASK_MODE = args.mask_mode
    

    
    # Adversarial audio settings
    config.ADVERSARIAL_AUG = args.adversarial_aug
    config.ADVERSARIAL_VARIANT = args.adversarial_variant

    # --- 2. Centralized Path Management (Now Chunk-Aware) ---
    experiment_name = args.experiment
    model_alias = config.MODEL_ALIAS
    
    # For audio_masking, jasco_masking, and adversarial, use hierarchical subdirs
    if experiment_name == 'audio_masking':
        output_dir = os.path.join(config.RESULTS_DIR, model_alias, experiment_name,
                                  config.MASK_TYPE, config.MASK_MODE)
    elif experiment_name == 'adversarial' and config.ADVERSARIAL_AUG:
        output_dir = os.path.join(config.RESULTS_DIR, model_alias, experiment_name,
                                  config.ADVERSARIAL_AUG)
    else:
        output_dir = os.path.join(config.RESULTS_DIR, model_alias, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # We start with the base filename.
    base_filename = f"{experiment_name}_{model_alias}_{args.dataset}"
    if config.RESTRICTED:
        base_filename += "-restricted"
    
    # Add suffix for external perturbation runs (to distinguish from self-perturbation)
    if config.USE_EXTERNAL_PERTURBATIONS:
        base_filename += "-mistral"
    
    # Add suffix for lorem filler type (to distinguish from dots filler results)
    if config.FILLER_TYPE == 'lorem':
        base_filename += "-lorem"
    
    # Add suffix for audio masking experiments (to create separate files per mask_type/mask_mode)
    if experiment_name == 'audio_masking':
        base_filename += f"_{config.MASK_TYPE}_{config.MASK_MODE}"
    
    # Add suffix for adversarial experiments (aug mode + variant)
    if config.ADVERSARIAL_AUG and config.ADVERSARIAL_VARIANT:
        base_filename += f"_{config.ADVERSARIAL_AUG}_{config.ADVERSARIAL_VARIANT}"
    
    # If this is a parallel run, we add the part number to the output filename.
    # e.g., 'adding_mistakes_salmonn_mmar-restricted.part_7.jsonl'
    if args.part is not None:
        output_filename = f"{base_filename}.part_{args.part}.jsonl"
    else:
        output_filename = f"{base_filename}.jsonl"
        
    config.OUTPUT_PATH = os.path.join(output_dir, output_filename)
    
    log_dir = os.path.join(config.RESULTS_DIR, 'logs')
    log_filepath = setup_logger(log_dir, model_alias, experiment_name, args.dataset)
    
    # --- 3. Dynamic Model Utility Loading ---
    logging.info(f"Loading utility module for model: {model_alias}")
    try:
        model_key = config.MODEL_ALIASES[model_alias]
        model_path = config.MODEL_PATHS[model_key]
        
        if model_alias == 'qwen':
            from core import qwen_utils as model_utils
        elif model_alias == 'flamingo':
            from core import audio_flamingo_utils as model_utils
        elif model_alias in ('salmonn', 'salmonn_7b'):
            from core import salmonn_utils as model_utils
        else:
            raise ImportError(f"No utility module defined for model alias '{model_alias}'")

    except (ImportError, KeyError) as e:
        logging.exception(f"Could not load utilities for model '{model_alias}'.")
        sys.exit(1)

    # --- 4. logging.info a "Run Summary" Banner ---
    logging.info("--- LALM Faithfulness Framework ---")
    logging.info(f"  - Model:      {model_alias.upper()}")
    logging.info(f"  - Experiment: {args.experiment}")
    logging.info(f"  - Dataset:    {args.dataset}")
    
    # The run mode logging is now more detailed.
    run_mode = "RESTRICTED" if config.RESTRICTED else "FULL DATASET"
    if args.part is not None:
        run_mode += f" (PARALLEL - Part {args.part}/{args.total_parts})"
    logging.info(f"  - Run Mode:   {run_mode}")
    
    logging.info(f"  - Outputting to: {config.OUTPUT_PATH}")
    logging.info(f"  - Verbose Logging: {'Enabled' if config.VERBOSE else 'Disabled'}")
    logging.info(f"  - Full log will be saved to: {log_filepath}")
    logging.info("-" * 35)

    # --- 5. Dynamic Experiment Loading ---
    # Experiments are now organized into subfolders: baseline/, cot/, audio_interventions/
    # We try each subfolder in order until we find the experiment module.
    EXPERIMENT_SUBFOLDERS = ['baseline', 'cot', 'audio_interventions']
    
    try:
        experiment_module = None
        exp_name = args.experiment
        
        # When using external perturbations, load the _combined version of the experiment
        # which expects pre-combined files from scripts/combine_baseline_with_perturbations.py
        if config.USE_EXTERNAL_PERTURBATIONS and exp_name in ['adding_mistakes', 'paraphrasing']:
            exp_name = f"{exp_name}_combined"
            logging.info(f"Using COMBINED experiment module: {exp_name}")
        
        # Try each subfolder in order
        for subfolder in EXPERIMENT_SUBFOLDERS:
            try:
                experiment_module_name = f"experiments.{subfolder}.{exp_name}"
                experiment_module = importlib.import_module(experiment_module_name)
                logging.info(f"Loaded experiment from: experiments/{subfolder}/{exp_name}.py")
                break
            except ImportError:
                continue
        
        if experiment_module is None:
            raise ImportError(f"Could not find experiment '{exp_name}' in any subfolder: {EXPERIMENT_SUBFOLDERS}")
        
        EXPERIMENT_TYPE = getattr(experiment_module, 'EXPERIMENT_TYPE')
    except (ImportError, AttributeError):
        logging.exception(f"Could not load experiment '{args.experiment}'.")
        sys.exit(1)

    logging.info(f"Detected experiment type: '{EXPERIMENT_TYPE}'")

    # --- GUARDRAIL 1: Check for num_chains > 0 ---
    if config.NUM_CHAINS_PER_QUESTION <= 0:
        logging.error("FATAL: NUM_CHAINS_PER_QUESTION must be a positive integer.")
        logging.error("Please set a value > 0 in config.py or use the --num-chains flag.")
        sys.exit(1)

    if EXPERIMENT_TYPE == "foundational":
        # --- GUARDRAIL 2 & 3 for Foundational Experiments ---
        if args.restricted:
            logging.error("FATAL: The --restricted flag cannot be used with foundational experiments.")
            logging.error("The 'restricted' dataset is created FROM foundational results, not the other way around.")
            logging.error("To create a restricted dataset, first run the 'baseline' and 'no_reasoning' experiments, then run:")
            logging.error(f"  python data_processing/create_restricted_dataset.py --model {model_alias} --dataset {args.dataset}")
            sys.exit(1)
        
        if args.part is not None:
            logging.error("FATAL: Parallelization with the --part flag is not supported for foundational experiments as of yet.")
            logging.error("To run a dependent experiment in parallel, you must first generate the full foundational results, then split them using:")
            logging.error(f"  python data_processing/split_dataset_for_parallel_runs.py --model {model_alias} --dataset {args.dataset} --num-parts <N>")
            sys.exit(1)
    
    
    # --- 6. Load the Model using the Dynamic Utilities ---
    logging.info("Loading model, processor and tokenizer...")
    model, processor, tokenizer = model_utils.load_model_and_tokenizer(model_path)

    # --- 7. Execute the Experiment Based on its Type ---
    if EXPERIMENT_TYPE == "foundational":
        # Foundational experiments are not parallelized at this level, so their logic is unchanged.
        logging.info("Running a FOUNDATIONAL experiment...")
        try:
            # For adversarial experiments, override dataset path to load adversarial JSONL
            if config.ADVERSARIAL_AUG and config.ADVERSARIAL_VARIANT:
                track = args.dataset.split('-')[1]  # e.g., 'sakura-animal' -> 'animal'
                dataset_path = os.path.join(
                    'data', 'adversarial_aug_data',
                    f"{track}_{config.ADVERSARIAL_AUG}",
                    f"adversarial_{track}_{config.ADVERSARIAL_AUG}_{config.ADVERSARIAL_VARIANT}.jsonl"
                )
                logging.info(f"[ADVERSARIAL] Using adversarial dataset: {dataset_path}")
            else:
                dataset_path = config.DATASET_MAPPING[args.dataset]
            
            data_samples = load_dataset(dataset_path)
            if config.NUM_SAMPLES_TO_RUN > 0:
                data_samples = data_samples[:config.NUM_SAMPLES_TO_RUN]
            logging.info(f"Processing {len(data_samples)} samples from '{dataset_path}'.")
            experiment_module.run(model, processor, tokenizer, model_utils, data_samples, config)
        except (KeyError, FileNotFoundError):
            logging.exception("Could not load dataset.")
            sys.exit(1)

    elif EXPERIMENT_TYPE == "dependent":
        logging.info("Running a DEPENDENT experiment...")
        
        # --- Centralized, Chunk-Aware Logic for Finding Dependency Files ---
        def get_dependency_path(exp_name):
            base_dir = os.path.join(config.RESULTS_DIR, model_alias, exp_name)
            filename = f"{exp_name}_{model_alias}_{args.dataset}"
            if config.RESTRICTED:
                filename += "-restricted"
            if args.part is not None:
                filename += f".part_{args.part}"
            filename += ".jsonl"
            return os.path.join(base_dir, filename)

        config.BASELINE_RESULTS_PATH = get_dependency_path('baseline')
        config.NO_REASONING_RESULTS_PATH = get_dependency_path('no_reasoning')
        
        experiment_module.run(model, processor, tokenizer, model_utils, config)

    elif EXPERIMENT_TYPE == "independent":
        # Independent experiments need baseline results for consistency comparison
        # but operate on their own data (e.g., masked audio datasets)
        logging.info("Running an INDEPENDENT experiment...")
        
        # Get baseline path for consistency comparison
        def get_dependency_path(exp_name):
            base_dir = os.path.join(config.RESULTS_DIR, model_alias, exp_name)
            filename = f"{exp_name}_{model_alias}_{args.dataset}"
            if config.RESTRICTED:
                filename += "-restricted"
            if args.part is not None:
                filename += f".part_{args.part}"
            filename += ".jsonl"
            return os.path.join(base_dir, filename)
        
        config.BASELINE_RESULTS_PATH = get_dependency_path('baseline')
        
        # Load original dataset samples
        try:
            dataset_path = config.DATASET_MAPPING[args.dataset]
            data_samples = load_dataset(dataset_path)
            if config.NUM_SAMPLES_TO_RUN > 0:
                data_samples = data_samples[:config.NUM_SAMPLES_TO_RUN]
            logging.info(f"Processing {len(data_samples)} samples from '{dataset_path}'.")
            experiment_module.run(model, processor, tokenizer, model_utils, data_samples, config)
        except (KeyError, FileNotFoundError):
            logging.exception("Could not load dataset.")
            sys.exit(1)

    else:
        logging.info(f"FATAL: Unknown experiment type '{EXPERIMENT_TYPE}'.")
        sys.exit(1)

    logging.info("--- Experiment completed successfully! ---")

if __name__ == "__main__":
    main()