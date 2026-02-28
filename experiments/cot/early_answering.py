# experiments/early_answering.py

"""
This script conducts the "Early Answering" intervention experiment.

Methodology:
1. It takes a sanitized baseline reasoning chain (CoT).
2. It reveals the chain progressively, from 0 sentences to N sentences.
3. At each step it runs a conditioned inference and records the answer.
4. This yields a lock-in curve showing how answer consistency evolves as more
   reasoning is provided.
"""

import os
import json
import collections
import nltk
import logging
from core.prompt_strategies import run_conditioned_trial, run_no_reasoning_trial

# This is a 'dependent' experiment because it relies on baseline CoTs.
EXPERIMENT_TYPE = "dependent"

PROMPT_PROTOCOL_BY_MODEL = {
    "qwen_omni": "qwen_omni_xml_conditioned",
    "flamingo_hf": "af3_conditioned_short_answer",
}


def _get_prompt_protocol(model_alias: str) -> str:
    return PROMPT_PROTOCOL_BY_MODEL.get(model_alias, "legacy_two_turn_conditioned")


def run_early_answering_trial(
    model,
    processor,
    tokenizer,
    model_utils,
    question: str,
    choices_formatted: str,
    audio_path: str,
    truncated_cot: str,
    num_sentences_provided: int,
) -> dict:
    """
    Runs a single early-answering trial.

    For num_sentences_provided == 0 (no reasoning given), uses the
    no-reasoning prompt variant so the model is not cued to reason.
    For num_sentences_provided > 0, uses conditioned inference with
    the truncated reasoning chain.
    """
    if num_sentences_provided == 0:
        result = run_no_reasoning_trial(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            model_utils=model_utils,
            question=question,
            choices=choices_formatted,
            audio_path=audio_path,
        )
    else:
        result = run_conditioned_trial(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            model_utils=model_utils,
            question=question,
            choices=choices_formatted,
            audio_path=audio_path,
            provided_reasoning=truncated_cot,
        )

    return {
        "question": question,
        "choices": choices_formatted,
        "audio_path": audio_path,
        "predicted_choice": result.get("predicted_choice"),
        "final_answer_raw": result.get("final_answer_raw", ""),
        "final_prompt_messages": result.get("final_prompt_messages", []),
    }


def run(model, processor, tokenizer, model_utils, config):
    """
    Orchestrates the full early answering experiment with a robust, model-agnostic,
    and restartable design.
    """
    # --- 1. Load Dependent Data (Centralized Pathing) ---
    baseline_results_path = config.BASELINE_RESULTS_PATH
    if not os.path.exists(baseline_results_path):
        logging.error(f"Baseline results file not found at: '{baseline_results_path}'")
        return

    logging.info(f"Reading baseline data from '{baseline_results_path}'...")
    all_baseline_trials = []
    with open(baseline_results_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                all_baseline_trials.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping corrupted baseline line {line_num}.")
                continue

    # --- 2. Prepare Data & Apply Filters ---
    if config.NUM_CHAINS_PER_QUESTION > 0:
        logging.info(f"Filtering to include only the first {config.NUM_CHAINS_PER_QUESTION} chains for each question.")
        all_baseline_trials = [
            trial for trial in all_baseline_trials
            if trial['chain_id'] < config.NUM_CHAINS_PER_QUESTION
        ]

    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials: trials_by_question[trial['id']].append(trial)
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # --- 3. Restartability Logic ---
    output_path = config.OUTPUT_PATH
    completed_steps = set()
    completed_chains = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    step_key = (data["id"], data["chain_id"], data["num_sentences_provided"])
                    completed_steps.add(step_key)
                    if data["num_sentences_provided"] == data["total_sentences_in_chain"]:
                        completed_chains.add((data["id"], data["chain_id"]))
                except (json.JSONDecodeError, KeyError): continue
    if completed_chains:
        logging.info(f"Found {len(completed_chains)} fully completed chains. They will be skipped.")

    # --- 4. Run the Experiment ---
    prompt_protocol = _get_prompt_protocol(config.MODEL_ALIAS)
    logging.info(
        f"Running Early Answering Intervention (Model: {config.MODEL_ALIAS.upper()}, "
        f"Protocol: {prompt_protocol}): Saving to {output_path}"
    )
    
    skipped_trials_count = 0
    processed_steps_count = 0
    with open(output_path, 'a') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if (q_id, chain_id) in completed_chains:
                    continue

                if config.VERBOSE:
                    logging.info(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                choices_formatted = baseline_trial['choices']
                sanitized_cot = baseline_trial.get("sanitized_cot")
                if sanitized_cot is None:
                    skipped_trials_count += 1
                    logging.warning(
                        f"Skipping ID {q_id}, chain {chain_id}: missing 'sanitized_cot' in baseline row."
                    )
                    continue

                sentences = nltk.sent_tokenize(sanitized_cot)
                total_sentences = len(sentences)

                for num_sentences_provided in range(total_sentences + 1):
                    step_key = (q_id, chain_id, num_sentences_provided)
                    if step_key in completed_steps:
                        continue

                    modified_reasoning = (
                        "" if num_sentences_provided == 0 else " ".join(sentences[:num_sentences_provided])
                    )

                    trial_result = run_early_answering_trial(
                        model, processor, tokenizer, model_utils, 
                        baseline_trial['question'], 
                        choices_formatted, 
                        baseline_trial['audio_path'], 
                        modified_reasoning,
                        num_sentences_provided,
                    )
                    # 
                    trial_result['id'] = q_id
                    trial_result['chain_id'] = chain_id
                    trial_result['num_sentences_provided'] = num_sentences_provided
                    trial_result['total_sentences_in_chain'] = total_sentences
                    trial_result['prompt_protocol'] = prompt_protocol
                    trial_result['sanitized_cot'] = sanitized_cot
                    trial_result['modified_reasoning'] = modified_reasoning
                    
                    trial_result['correct_choice'] = baseline_trial['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == trial_result['correct_choice'])
                    trial_result['corresponding_baseline_predicted_choice'] = baseline_trial['predicted_choice']
                    trial_result['is_consistent_with_baseline'] = (trial_result['predicted_choice'] == baseline_trial['predicted_choice'])
                    
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "chain_id": trial_result['chain_id'],
                        "num_sentences_provided": trial_result['num_sentences_provided'],
                        "total_sentences_in_chain": trial_result['total_sentences_in_chain'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
                        "corresponding_baseline_predicted_choice": trial_result['corresponding_baseline_predicted_choice'],
                        "is_consistent_with_baseline": trial_result['is_consistent_with_baseline'],
                        "final_answer_raw": trial_result['final_answer_raw'],
                        "modified_reasoning": trial_result["modified_reasoning"],
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "sanitized_cot": trial_result["sanitized_cot"],
                        "audio_path": trial_result['audio_path'],
                        "question": trial_result['question'],
                        "choices": trial_result['choices'],
                        "model": config.MODEL_ALIAS,
                        "prompt_protocol": trial_result["prompt_protocol"],
                    }
                    
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    f.flush()
                    completed_steps.add(step_key)
                    processed_steps_count += 1

            except Exception as e:
                skipped_trials_count += 1
                logging.exception(f"SKIPPING ENTIRE TRIAL due to unhandled error. ID: {baseline_trial.get('id', 'N/A')}, Chain: {baseline_trial.get('chain_id', 'N/A')}")
                continue

    # Final summary
    total_processed_in_this_run = len(samples_to_process) - len(completed_chains)
    logging.info(f"--- Early Answering experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total trials in dataset: {len(samples_to_process)}")
    logging.info(f"Trials already complete: {len(completed_chains)}")
    logging.info(f"Chains touched in this run: {total_processed_in_this_run - skipped_trials_count}")
    logging.info(f"Inference steps processed in this run: {processed_steps_count}")
    logging.info(f"Skipped trials due to errors in this run: {skipped_trials_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)
