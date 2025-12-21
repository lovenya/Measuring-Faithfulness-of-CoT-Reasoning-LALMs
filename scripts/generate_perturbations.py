#!/usr/bin/env python3
# scripts/generate_perturbations.py

"""
Pre-generate perturbations (mistakes or paraphrases) using Mistral Small 3.

This script is Phase A of the two-phase external perturbation workflow:
  Phase A: Run this script with Mistral to generate all perturbations, save to file
  Phase B: Run experiment scripts with --use-external-perturbations flag

Usage:
    # For mistakes:
    python scripts/generate_perturbations.py \
        --baseline-results results/baseline/qwen_mmar.jsonl \
        --output results/mistral_perturbations/qwen_mmar_mistakes.jsonl \
        --mode mistakes

    # For paraphrasing:
    python scripts/generate_perturbations.py \
        --baseline-results results/baseline/qwen_mmar.jsonl \
        --output results/mistral_perturbations/qwen_mmar_paraphrases.jsonl \
        --mode paraphrase
"""

import argparse
import json
import logging
import os
import sys
import nltk
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# --- Environment Setup: NLTK Data Path ---
# Use the local nltk_data directory bundled with the project
local_nltk_data_path = os.path.join(PROJECT_ROOT, 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.insert(0, local_nltk_data_path)

from core.mistral_utils import load_mistral_model, generate_mistake, paraphrase_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_baseline_results(baseline_path: str) -> list[dict]:
    """Load baseline results from JSONL file."""
    results = []
    with open(baseline_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping corrupted line {line_num}: {e}")
    return results


def load_existing_perturbations(output_path: str) -> set:
    """Load already-generated perturbations for restartability."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Key: (id, chain_id, sentence_idx)
                    completed.add((data['id'], data['chain_id'], data['sentence_idx']))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def generate_mistakes_for_baseline(model, tokenizer, baseline_results: list[dict], output_path: str):
    """Generate mistake perturbations for all sentences in baseline results."""
    
    completed = load_existing_perturbations(output_path)
    logger.info(f"Found {len(completed)} existing perturbations. Will skip these.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'a') as f:
        for trial in tqdm(baseline_results, desc="Generating mistakes"):
            q_id = trial['id']
            chain_id = trial['chain_id']
            question = trial['question']
            choices = trial['choices']
            sanitized_cot = trial.get('sanitized_cot', '')
            
            if not sanitized_cot:
                continue
            
            sentences = nltk.sent_tokenize(sanitized_cot)
            
            for sentence_idx, original_sentence in enumerate(sentences):
                # Skip if already done
                if (q_id, chain_id, sentence_idx) in completed:
                    continue
                
                # Skip short/meaningless sentences
                if len(original_sentence.split()) < 3:
                    continue
                
                # Generate mistake using Mistral
                mistaken = generate_mistake(model, tokenizer, question, choices, original_sentence)
                
                if mistaken is None:
                    logger.warning(f"Failed to generate mistake for {q_id}/{chain_id}/{sentence_idx}")
                    mistaken = ""  # Save empty to mark as attempted
                
                # Save result
                result = {
                    "id": q_id,
                    "chain_id": chain_id,
                    "sentence_idx": sentence_idx,
                    "original_sentence": original_sentence,
                    "mistaken_sentence": mistaken
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


def generate_paraphrases_for_baseline(model, tokenizer, baseline_results: list[dict], output_path: str):
    """Generate paraphrase perturbations for progressive paraphrasing."""
    
    completed = load_existing_perturbations(output_path)
    logger.info(f"Found {len(completed)} existing perturbations. Will skip these.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'a') as f:
        for trial in tqdm(baseline_results, desc="Generating paraphrases"):
            q_id = trial['id']
            chain_id = trial['chain_id']
            sanitized_cot = trial.get('sanitized_cot', '')
            
            if not sanitized_cot:
                continue
            
            sentences = nltk.sent_tokenize(sanitized_cot)
            
            # For paraphrasing, we paraphrase progressively (1 sentence, 2 sentences, etc.)
            for num_to_paraphrase in range(1, len(sentences) + 1):
                # Use num_to_paraphrase as sentence_idx for lookup key
                if (q_id, chain_id, num_to_paraphrase) in completed:
                    continue
                
                # Paraphrase first N sentences
                text_to_paraphrase = " ".join(sentences[:num_to_paraphrase])
                paraphrased = paraphrase_text(model, tokenizer, text_to_paraphrase)
                
                if paraphrased is None:
                    logger.warning(f"Failed to paraphrase for {q_id}/{chain_id}/n={num_to_paraphrase}")
                    paraphrased = ""
                
                result = {
                    "id": q_id,
                    "chain_id": chain_id,
                    "sentence_idx": num_to_paraphrase,  # num_sentences_paraphrased
                    "original_text": text_to_paraphrase,
                    "paraphrased_text": paraphrased
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate perturbations using Mistral Small 3"
    )
    parser.add_argument(
        "--baseline-results", "-b",
        type=str,
        required=True,
        help="Path to baseline results JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for perturbations JSONL file"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["mistakes", "paraphrase"],
        required=True,
        help="Type of perturbation to generate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/mistral-small-3",
        help="Path to Mistral model weights"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Limit to first N baseline trials (for testing)"
    )
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.baseline_results):
        logger.error(f"Baseline results not found: {args.baseline_results}")
        return 1
    
    # Load baseline
    logger.info(f"Loading baseline results from: {args.baseline_results}")
    baseline_results = load_baseline_results(args.baseline_results)
    
    # Limit samples if specified
    if args.num_samples is not None:
        baseline_results = baseline_results[:args.num_samples]
        logger.info(f"Limited to {len(baseline_results)} baseline trials (--num-samples)")
    else:
        logger.info(f"Loaded {len(baseline_results)} baseline trials")
    
    # Load model
    logger.info("Loading Mistral Small 3...")
    model, tokenizer = load_mistral_model(model_path=args.model_path)
    
    # Generate perturbations
    if args.mode == "mistakes":
        logger.info(f"Generating mistake perturbations -> {args.output}")
        generate_mistakes_for_baseline(model, tokenizer, baseline_results, args.output)
    else:
        logger.info(f"Generating paraphrase perturbations -> {args.output}")
        generate_paraphrases_for_baseline(model, tokenizer, baseline_results, args.output)
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
