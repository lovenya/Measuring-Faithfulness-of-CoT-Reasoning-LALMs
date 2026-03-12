#!/usr/bin/env python3
"""
JASCO Evaluation Script (Stage 2): LLM-as-a-Judge

Reads Stage 1 JSONL output from jasco_masking.py and uses an LLM judge
(default: Mistral Small 3 via vLLM) to score each model prediction on a
0-2 scale using the JASCO Task 1 rubric.

Input path is auto-built:  results/{model}/jasco_masking/jasco_masking_{model}_jasco.jsonl
Output path is auto-built: results/{model}/jasco_masking/llm_judge_evaluations/jasco_masking_{model}_jasco_evaluated_by_{judge}.jsonl

Usage:
    python analysis/evaluate_jasco.py --model qwen
    python analysis/evaluate_jasco.py --model qwen --judge mistral
    python analysis/evaluate_jasco.py --model salmonn --judge mistral
"""

import argparse
import json
import os
import sys
import re
import time
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.mistral_utils import load_mistral_model, run_mistral_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# The JASCO Task 1 judge prompt template
# NOTE: The final instruction explicitly asks for "Score: X" to ensure parseable output.
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
Score 0: The speaker's action predicted is completely misaligned, providing incorrect or irrelevant information compared to the speaker's action in the reference or the response is too general such as 'talking to someone' or 'having conversation'
Score 1: The speaker's action predicted aligns with the speaker's action in the reference generally but lacks detailed keywords, the predicted action is logical enough but not the most possible.
Score 2: The speaker's action predicted is highly accurate, and matches the speaker's action in the reference perfectly, capturing its essence and detailed keywords. The prediction is very logical and the most probable.

After your analysis, you MUST end your response with exactly one of the following on its own line:
Score: 0
Score: 1
Score: 2"""

# The question is always the same for all samples
JASCO_QUESTION = (
    "Based on both the spoken text and the audio sound, "
    "think step by step and infer what the speakers are doing specifically?"
)


def parse_score(judge_response: str) -> int | None:
    """
    Parse the judge's score from the LAST LINE of its response.

    We only look at the last non-empty line to avoid false matches
    from the analysis body (e.g., "Score 0 criteria says..." ).
    """
    if not judge_response:
        return None

    # Get the last non-empty line
    lines = [line.strip() for line in judge_response.strip().split('\n') if line.strip()]
    if not lines:
        return None

    last_line = lines[-1]

    # Try patterns on the last line only
    patterns = [
        r'[Ss]core\s*:\s*\**\s*([012])\s*\**',    # Score: 2, Score: **2**, **Score: 1**
        r'\*\*[Ss]core\s*:\s*([012])\*\*',          # **Score: 2**
        r'[Ss]core\s*:\s*([012])',                   # Score: 2, score: 1
        r'[Ss]core\s*([012])\b',                     # Score2, Score 1
        r'\b([012])\s*/\s*2\b',                      # 2/2, 1/2
    ]

    for pattern in patterns:
        match = re.search(pattern, last_line, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fallback: check second-to-last line too (sometimes there's trailing whitespace/formatting)
    if len(lines) >= 2:
        second_last = lines[-2]
        for pattern in patterns:
            match = re.search(pattern, second_last, re.IGNORECASE)
            if match:
                return int(match.group(1))

    return None


def evaluate_single(model, entry: dict) -> dict:
    """
    Use the LLM judge to score a single Stage 1 prediction.
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
        max_new_tokens=512,
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
        description="JASCO Stage 2: LLM-as-a-Judge evaluation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Model alias (e.g., 'qwen', 'salmonn', 'salmonn_7b').\n"
                             "Input path is auto-built: results/{model}/jasco_masking/jasco_masking_{model}_jasco.jsonl")
    parser.add_argument('--judge', type=str, default='mistral',
                        choices=['mistral'],
                        help="Which LLM judge to use (default: mistral).\n"
                             "  mistral = Mistral Small 3 via vLLM")
    parser.add_argument('--results-dir', type=str, default='results',
                        help="Base results directory (default: results).")
    parser.add_argument('--mistral-path', type=str,
                        default=config.MODEL_PATHS.get('mistral_small_3'),
                        help="Path to Mistral Small 3 weights.")
    args = parser.parse_args()

    # Build input path automatically
    input_path = os.path.join(
        args.results_dir, args.model, 'jasco_masking',
        f'jasco_masking_{args.model}_jasco.jsonl'
    )

    # Build output path: results/{model}/jasco_masking/llm_judge_evaluations/...evaluated_by_{judge}.jsonl
    eval_dir = os.path.join(
        args.results_dir, args.model, 'jasco_masking', 'llm_judge_evaluations'
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_path = os.path.join(
        eval_dir,
        f'jasco_masking_{args.model}_jasco_evaluated_by_{args.judge}.jsonl'
    )

    logger.info(f"Model:  {args.model}")
    logger.info(f"Judge:  {args.judge}")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    # Validate input exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run the JASCO masking experiment (Stage 1) first.")
        sys.exit(1)

    # Load Stage 1 results
    logger.info(f"Loading Stage 1 results from: {input_path}")
    with open(input_path, 'r') as f:
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
        # Load judge model
        logger.info(f"Loading judge model: {args.judge}")
        if args.judge == 'mistral':
            judge_model = load_mistral_model(args.mistral_path)
        else:
            raise ValueError(f"Unknown judge: {args.judge}")

        # Score each entry with progress bar
        null_count = 0
        scored_count = 0
        start_time = time.time()

        with open(output_path, 'a') as f:
            pbar = tqdm(
                entries_to_score,
                desc="Scoring",
                unit="entry",
                dynamic_ncols=True,
            )
            for i, entry in enumerate(pbar):
                try:
                    scored_entry = evaluate_single(judge_model, entry)
                except Exception as e:
                    logger.error(f"Error scoring entry {entry['id']} / {entry['condition']}: {e}")
                    scored_entry = {**entry, 'judge_response': f"ERROR: {e}", 'judge_score': None}

                f.write(json.dumps(scored_entry) + '\n')
                f.flush()

                if scored_entry.get('judge_score') is not None:
                    scored_count += 1
                else:
                    null_count += 1

                # Update progress bar with timing info
                elapsed = time.time() - start_time
                avg_per_entry = elapsed / (i + 1)
                remaining = avg_per_entry * (len(entries_to_score) - i - 1)

                pbar.set_postfix({
                    'ok': scored_count,
                    'null': null_count,
                    'avg': f'{avg_per_entry:.1f}s',
                    'ETA': f'{remaining/60:.1f}m',
                })

        total_elapsed = time.time() - start_time
        logger.info(f"Scoring complete in {total_elapsed/60:.1f} minutes "
                     f"({total_elapsed/len(entries_to_score):.1f}s/entry)")
        logger.info(f"Successfully parsed: {scored_count}, Failed to parse: {null_count}")

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
