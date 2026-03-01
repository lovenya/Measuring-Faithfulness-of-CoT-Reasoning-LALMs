#!/usr/bin/env python3
# scripts/combine_baseline_with_perturbations.py

"""
Preprocessing script to combine baseline results with Mistral-generated perturbations.

This creates "combined" files that have all the data needed for each experiment trial,
making them ready for splitting and parallel processing without on-the-fly lookup.

Output Structure:
-----------------
For adding_mistakes:
    Each line = one trial (one mistake position in one chain)
    {
        "id": "mmar_0",
        "chain_id": 0,
        "mistake_position": 2,  # 1-indexed (position where mistake is inserted)
        "total_sentences": 5,
        "prefix_sentences": ["sent_0", "sent_1"],  # unchanged baseline sentences before mistake
        "mistaken_sentence": "...",  # from Mistral
        "question": "...",
        "choices": "...",
        "audio_path": "...",
        "correct_choice": "A",
        "baseline_predicted_choice": "B",
        # Note: NO remaining sentences - model will continue generating from mistake
    }

For paraphrasing:
    Each line = one trial (one paraphrase level in one chain)
    {
        "id": "mmar_0",
        "chain_id": 0,
        "num_sentences_paraphrased": 3,  # how many sentences were paraphrased
        "total_sentences": 5,
        "paraphrased_text": "...",  # first N sentences paraphrased by Mistral
        "remaining_sentences": ["sent_3", "sent_4"],  # unchanged baseline sentences after
        "question": "...",
        "choices": "...",
        "audio_path": "...",
        "correct_choice": "A",
        "baseline_predicted_choice": "B",
    }

Usage:
------
    python scripts/combine_baseline_with_perturbations.py \\
        --model qwen \\
        --dataset mmar \\
        --experiment adding_mistakes \\
        --restricted

    # Or for all models/datasets:
    python scripts/combine_baseline_with_perturbations.py --all --experiment adding_mistakes
"""

import argparse
import json
import os
import sys
import logging
import nltk

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup local NLTK data path (same as main.py)
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.error(f"FATAL: NLTK 'punkt' model not found in '{local_nltk_data_path}'.")
        sys.exit(1)
else:
    logger.error(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    sys.exit(1)

# Constants
MODELS = ["qwen", "qwen_omni", "salmonn", "salmonn_7b", "salmonn_13b", "flamingo_hf"]
DATASETS = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]
EXPERIMENTS = ["adding_mistakes", "paraphrasing"]

# Paths
RESULTS_DIR = "results"
EXTERNAL_LLM_DIR = os.path.join(RESULTS_DIR, "external_llm_perturbations", "mistral")


def get_baseline_path(model: str, dataset: str, restricted: bool) -> str:
    """Get path to baseline results file."""
    suffix = "-restricted" if restricted else ""
    return os.path.join(RESULTS_DIR, model, "baseline", f"baseline_{model}_{dataset}{suffix}.jsonl")


def get_perturbation_path(model: str, dataset: str, experiment: str, restricted: bool) -> str:
    """Get path to Mistral perturbation file in the new directory structure."""
    if experiment == "adding_mistakes":
        pert_suffix = "mistakes"
    else:
        pert_suffix = "paraphrased"
    return os.path.join(EXTERNAL_LLM_DIR, model, dataset, "raw", f"{pert_suffix}.jsonl")


def get_output_path(model: str, dataset: str, experiment: str, restricted: bool) -> str:
    """Get path for combined output file in the new directory structure."""
    return os.path.join(EXTERNAL_LLM_DIR, model, dataset, "combined", f"{experiment}_combined.jsonl")


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num} in {path}: {e}")
    return data


def load_perturbations_as_dict(path: str, experiment: str) -> dict:
    """
    Load perturbations into a lookup dictionary.
    
    For adding_mistakes: key = (id, chain_id, sentence_idx) -> mistaken_sentence
    For paraphrasing: key = (id, chain_id, num_paraphrased) -> paraphrased_text
    """
    perturbations = {}
    data = load_jsonl(path)
    
    for item in data:
        key = (item['id'], item['chain_id'], item['sentence_idx'])
        if experiment == "adding_mistakes":
            perturbations[key] = item['mistaken_sentence']
        else:  # paraphrasing
            perturbations[key] = item['paraphrased_text']
    
    logger.info(f"Loaded {len(perturbations)} perturbations from {path}")
    return perturbations


def combine_for_adding_mistakes(
    baseline_data: list[dict],
    perturbations: dict,
    output_path: str
) -> tuple[int, int]:
    """
    Create combined file for adding_mistakes experiment.
    
    Returns: (num_written, num_skipped)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_written = 0
    num_skipped = 0
    
    with open(output_path, 'w') as f:
        for trial in baseline_data:
            q_id = trial['id']
            chain_id = trial['chain_id']
            sanitized_cot = trial['sanitized_cot']
            sentences = nltk.sent_tokenize(sanitized_cot)
            total_sentences = len(sentences)
            
            if total_sentences == 0:
                num_skipped += 1
                continue
            
            # For each possible mistake position
            for mistake_idx in range(total_sentences):
                original_sentence = sentences[mistake_idx]
                
                # Skip meaningless sentences (less than 3 words)
                if len(original_sentence.split()) < 3:
                    continue
                
                lookup_key = (q_id, chain_id, mistake_idx)
                mistaken_sentence = perturbations.get(lookup_key)
                
                if not mistaken_sentence:
                    num_skipped += 1
                    continue
                
                combined_record = {
                    "id": q_id,
                    "chain_id": chain_id,
                    "mistake_position": mistake_idx + 1,  # 1-indexed for consistency
                    "total_sentences": total_sentences,
                    "prefix_sentences": sentences[:mistake_idx],
                    "original_sentence": original_sentence,
                    "mistaken_sentence": mistaken_sentence,
                    "question": trial['question'],
                    "choices": trial['choices'],
                    "audio_path": trial['audio_path'],
                    "correct_choice": trial['correct_choice'],
                    "baseline_predicted_choice": trial['predicted_choice'],
                }
                
                f.write(json.dumps(combined_record, ensure_ascii=False) + "\n")
                num_written += 1
    
    return num_written, num_skipped


def combine_for_paraphrasing(
    baseline_data: list[dict],
    perturbations: dict,
    output_path: str
) -> tuple[int, int]:
    """
    Create combined file for paraphrasing experiment.
    
    Returns: (num_written, num_skipped)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_written = 0
    num_skipped = 0
    
    with open(output_path, 'w') as f:
        for trial in baseline_data:
            q_id = trial['id']
            chain_id = trial['chain_id']
            sanitized_cot = trial['sanitized_cot']
            sentences = nltk.sent_tokenize(sanitized_cot)
            total_sentences = len(sentences)
            
            if total_sentences == 0:
                num_skipped += 1
                continue
            
            # For each paraphrase level (1 to total_sentences)
            for num_paraphrased in range(1, total_sentences + 1):
                lookup_key = (q_id, chain_id, num_paraphrased)
                paraphrased_text = perturbations.get(lookup_key)
                
                if not paraphrased_text:
                    num_skipped += 1
                    continue
                
                remaining_sentences = sentences[num_paraphrased:]
                
                combined_record = {
                    "id": q_id,
                    "chain_id": chain_id,
                    "num_sentences_paraphrased": num_paraphrased,
                    "total_sentences": total_sentences,
                    "paraphrased_text": paraphrased_text,
                    "remaining_sentences": remaining_sentences,
                    "question": trial['question'],
                    "choices": trial['choices'],
                    "audio_path": trial['audio_path'],
                    "correct_choice": trial['correct_choice'],
                    "baseline_predicted_choice": trial['predicted_choice'],
                }
                
                f.write(json.dumps(combined_record, ensure_ascii=False) + "\n")
                num_written += 1
    
    return num_written, num_skipped


def process_combination(model: str, dataset: str, experiment: str, restricted: bool) -> bool:
    """
    Process one model/dataset/experiment combination.
    
    Returns True if successful, False otherwise.
    """
    baseline_path = get_baseline_path(model, dataset, restricted)
    perturbation_path = get_perturbation_path(model, dataset, experiment, restricted)
    output_path = get_output_path(model, dataset, experiment, restricted)
    
    # Check files exist
    if not os.path.exists(baseline_path):
        logger.error(f"Baseline not found: {baseline_path}")
        return False
    
    if not os.path.exists(perturbation_path):
        logger.error(f"Perturbation file not found: {perturbation_path}")
        return False
    
    logger.info(f"Processing: {model}/{dataset}/{experiment}")
    logger.info(f"  Baseline: {baseline_path}")
    logger.info(f"  Perturbations: {perturbation_path}")
    logger.info(f"  Output: {output_path}")
    
    # Load data
    baseline_data = load_jsonl(baseline_path)
    perturbations = load_perturbations_as_dict(perturbation_path, experiment)
    
    logger.info(f"  Loaded {len(baseline_data)} baseline trials")
    
    # Combine
    if experiment == "adding_mistakes":
        num_written, num_skipped = combine_for_adding_mistakes(baseline_data, perturbations, output_path)
    else:
        num_written, num_skipped = combine_for_paraphrasing(baseline_data, perturbations, output_path)
    
    logger.info(f"  ✓ Written: {num_written} records, Skipped: {num_skipped}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Combine baseline results with Mistral perturbations into single files."
    )
    parser.add_argument("--model", type=str, choices=MODELS, help="Model name")
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Dataset name")
    parser.add_argument("--experiment", type=str, choices=EXPERIMENTS, required=True,
                        help="Experiment type")
    parser.add_argument("--restricted", action="store_true", default=False,
                        help="Use restricted datasets (default: False)")
    parser.add_argument("--all", action="store_true",
                        help="Process all models and datasets")
    
    args = parser.parse_args()
    
    if args.all:
        # Process all combinations
        success_count = 0
        fail_count = 0
        for model in MODELS:
            for dataset in DATASETS:
                if process_combination(model, dataset, args.experiment, args.restricted):
                    success_count += 1
                else:
                    fail_count += 1
        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"Successful: {success_count}, Failed: {fail_count}")
    else:
        if not args.model or not args.dataset:
            parser.error("--model and --dataset are required unless --all is specified")
        process_combination(args.model, args.dataset, args.experiment, args.restricted)


if __name__ == "__main__":
    main()
