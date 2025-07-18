#!/usr/bin/env python3

import argparse
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.lalm_utils import load_model_and_tokenizer
from data_loader.data_loader import load_dataset, get_dataset_info
from experiments import baseline


def main():
    parser = argparse.ArgumentParser(description="Run LALM faithfulness experiments")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Path to the dataset JSONL file (e.g., 'mmar_test_standardized.jsonl')"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["baseline"],
        default="baseline",
        help="Which experiment to run"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override the model path from config.py"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override the number of samples to process from config.py"
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=None,
        help="Override the number of chains per question from config.py"
    )
    
    args = parser.parse_args()
    
    # Override config values if provided
    if args.model_path:
        config.MODEL_PATH = args.model_path
    if args.num_samples:
        config.NUM_SAMPLES_TO_RUN = args.num_samples
    if args.num_chains:
        config.NUM_CHAINS_PER_QUESTION = args.num_chains
    
    # Set dataset name from the dataset file path
    dataset_filename = os.path.basename(args.dataset)
    config.DATASET_NAME = dataset_filename.replace('.jsonl', '').replace('_test_standardized', '')
    
    print(f"=== LALM Faithfulness Experiment ===")
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.experiment}")
    print(f"Model Path: {config.MODEL_PATH}")
    print(f"Number of samples: {config.NUM_SAMPLES_TO_RUN}")
    print(f"Chains per question: {config.NUM_CHAINS_PER_QUESTION}")
    print(f"Results directory: {config.RESULTS_DIR}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    try:
        data_samples = load_dataset(args.dataset)
        dataset_info = get_dataset_info(data_samples)
        print(f"Dataset loaded successfully: {dataset_info}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Limit samples if specified
    if config.NUM_SAMPLES_TO_RUN > 0:
        data_samples = data_samples[:config.NUM_SAMPLES_TO_RUN]
        print(f"Limited to {len(data_samples)} samples")
    
    # Load model and processor
    print("\nLoading model and processor...")
    try:
        model, processor = load_model_and_tokenizer(config.MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run experiment
    print(f"\nRunning {args.experiment} experiment...")
    try:
        if args.experiment == "baseline":
            baseline.run(model, processor, data_samples, config)
        else:
            print(f"Unknown experiment: {args.experiment}")
            sys.exit(1)
    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()