#!/usr/bin/env python3
"""
JASCO Evaluation Script (Stage 2): LLM-as-a-Judge

Reads Stage 1 JSONL output from jasco_masking.py and uses Mistral Small 3
(via vLLM) to score each model prediction on a 0-2 scale using the JASCO
Task 1 rubric.

Output:
  - A scored JSONL file with the judge's score and reasoning appended.
  - Summary statistics printed to stdout (mean score per condition).

Usage:
    python analysis/evaluate_jasco.py --input results/qwen/jasco_masking/jasco_masking_qwen_jasco.jsonl
    python analysis/evaluate_jasco.py --input results/qwen/jasco_masking/jasco_masking_qwen_jasco.jsonl --output results/qwen/jasco_masking/scored.jsonl
"""

import argparse
import json
import os
import sys
import re
import logging
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.mistral_utils import load_mistral_model, run_mistral_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# The JASCO Task 1 judge prompt template
JUDGE_PROMPT_TEMPLATE = """[Question]
{question}


[Reference Answer]
{reference}

[Reference Answer Key Words]
{keywords}

[Model Prediction]
{prediction}


[Task1]
I am using a model to predict what the speakers are possibly doing based on both the audio sound and the spoken text. I want you to help me rate a score for the model's prediction on the speaker's action given a list of information [Question, Reference Answer, Reference Answer Key Words, Model Prediction]
Criteria: Assess if the model's prediction of the speaker's action mirrors the speaker's action in the reference answer in terms of content, logic, and relevance. Also assess if the model's prediction contains the Reference Answer Key Words or similar meanings. Do not care about the verb tense or other useless details in the model's response, focus only on the parts that speaker's actions are mentioned and the keywords. Very important: if the response does not create a prediction of the speaker's specific action, rate it directly 0, an example prediction like this can be 'The audio clip contains the sound of [some sounds]. The speaker says [some spoken texts]''.
Score0: The speaker's action predicted is completely misaligned, providing incorrect or irrelevant information compared to the speaker's action in the reference or the response is too general such as 'talking to someone' or 'having conversation'
Score1: The speaker's action predicted aligns with the speaker's action in the reference generally but lacks detailed keywords, the predicted action is logical enough but not the most possible.
Score2: The speaker's action predicted is highly accurate, and matches the speaker's action in the reference perfectly, capturing its essence and detailed keywords. The prediction is very logical and the most probable."""

# The question is always the same for all samples
JASCO_QUESTION = (
    "Based on both the spoken text and the audio sound, "
    "think step by step and infer what the speakers are doing specifically?"
)


def parse_score(judge_response: str) -> int | None:
    """
    Parse the judge's score from its response.

    Looks for patterns like 'Score: 2', 'Score2', 'score 1', 'Score: 0', etc.
    Returns the score as an integer (0, 1, or 2), or None if parsing fails.
    """
    if not judge_response:
        return None

    # Try multiple patterns, from most specific to least
    patterns = [
        r'[Ss]core\s*:\s*([012])',   # Score: 2, score: 1
        r'[Ss]core\s*([012])',        # Score2, Score 1
        r'\b([012])\s*/\s*2\b',       # 2/2, 1/2
    ]

    for pattern in patterns:
        matches = re.findall(pattern, judge_response)
        if matches:
            # Take the last match (the final verdict)
            return int(matches[-1])

    return None


def evaluate_single(model, entry: dict) -> dict:
    """
    Use Mistral to score a single Stage 1 prediction.
    Returns the entry augmented with judge_response and judge_score.
    """
    keywords_str = ', '.join(entry.get('target_keywords', []))

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=JASCO_QUESTION,
        reference=entry.get('correct_answer', ''),
        keywords=keywords_str,
        prediction=entry.get('model_output', ''),
    )

    messages = [{"role": "user", "content": prompt}]

    judge_response = run_mistral_inference(
        model, messages,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
    )

    score = parse_score(judge_response)

    return {
        **entry,
        'judge_response': judge_response,
        'judge_score': score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="JASCO Stage 2: LLM-as-a-Judge evaluation using Mistral."
    )
    parser.add_argument('--input', type=str, required=True,
                        help="Path to Stage 1 JSONL results file.")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to output scored JSONL (default: input with '_scored' suffix).")
    parser.add_argument('--mistral-path', type=str,
                        default=config.MODEL_PATHS.get('mistral_small_3'),
                        help="Path to Mistral Small 3 weights.")
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_scored{ext}"

    # Load Stage 1 results
    logger.info(f"Loading Stage 1 results from: {args.input}")
    with open(args.input, 'r') as f:
        entries = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(entries)} entries.")

    # --- Restartability: skip already-scored entries ---
    completed_keys = set()
    if os.path.exists(output_path):
        logger.info(f"Found existing scored file. Checking for completed entries...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_keys.add((data['id'], data['condition'], data['chain_id']))
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Found {len(completed_keys)} already-scored entries. They will be skipped.")

    entries_to_score = [
        e for e in entries
        if (e['id'], e['condition'], e['chain_id']) not in completed_keys
    ]
    logger.info(f"Entries to score: {len(entries_to_score)}")

    if not entries_to_score:
        logger.info("Nothing to score. All entries already evaluated.")
    else:
        # Load Mistral
        logger.info(f"Loading Mistral Small 3 from: {args.mistral_path}")
        mistral_model = load_mistral_model(args.mistral_path)

        # Score each entry
        with open(output_path, 'a') as f:
            for i, entry in enumerate(entries_to_score):
                try:
                    scored_entry = evaluate_single(mistral_model, entry)
                except Exception as e:
                    logger.error(f"Error scoring entry {entry['id']} / {entry['condition']}: {e}")
                    scored_entry = {**entry, 'judge_response': f"ERROR: {e}", 'judge_score': None}

                f.write(json.dumps(scored_entry) + '\n')
                f.flush()

                if (i + 1) % 50 == 0 or i == len(entries_to_score) - 1:
                    logger.info(f"Scored {i + 1}/{len(entries_to_score)}")

    # --- Summary Statistics ---
    logger.info("\n--- Summary Statistics ---")

    # Reload all scored entries
    all_scored = []
    with open(output_path, 'r') as f:
        for line in f:
            try:
                all_scored.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Group scores by condition
    scores_by_condition = defaultdict(list)
    for entry in all_scored:
        if entry.get('judge_score') is not None:
            scores_by_condition[entry['condition']].append(entry['judge_score'])

    # Define condition order for nice printing
    condition_order = ['baseline']
    for pct in range(10, 101, 10):
        condition_order.append(f'speech_{pct}')
    for pct in range(10, 101, 10):
        condition_order.append(f'audio_{pct}')

    print(f"\n{'Condition':<20} {'Count':>6} {'Mean Score':>12} {'Score Distribution':>20}")
    print("-" * 65)
    for cond in condition_order:
        scores = scores_by_condition.get(cond, [])
        if scores:
            mean_score = sum(scores) / len(scores)
            dist = {s: scores.count(s) for s in [0, 1, 2]}
            print(f"{cond:<20} {len(scores):>6} {mean_score:>12.3f} "
                  f"  0:{dist[0]:>3} 1:{dist[1]:>3} 2:{dist[2]:>3}")
        else:
            print(f"{cond:<20} {'N/A':>6}")

    failed = sum(1 for e in all_scored if e.get('judge_score') is None)
    if failed:
        print(f"\nWARNING: {failed} entries failed to parse a score.")

    logger.info(f"Scored results saved to: {output_path}")


if __name__ == '__main__':
    main()
