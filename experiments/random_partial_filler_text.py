# experiments/random_partial_filler_text.py

"""
This script conducts the "Random Partial Filler Text" experiment. This is a
variant of the filler text experiments designed to test if the position of
corrupted reasoning matters.

Methodology:
1. It takes a reasoning chain (CoT) from a baseline run.
2. For a given percentage (e.g., 20%), it replaces that percentage of the
   *words* in the CoT with a filler token ("..."). The words to be replaced
   are chosen randomly.
3. It then presents this partially corrupted CoT to the model and asks for a
   final answer.
4. By comparing the results to the "start" and "end" corruption variants, we
   can determine if the model is more sensitive to errors at the beginning or
   end of its reasoning chain.
"""

import os
import json
import collections
import random
import logging
from core.filler_text_utils import create_word_level_masked_cot, run_filler_trial

# This is a 'dependent' experiment because it manipulates the CoTs from a 'baseline' run.
EXPERIMENT_TYPE = "dependent"

def run(model, processor, tokenizer, model_utils, config):
    """
    Orchestrates the WORD-LEVEL RANDOM partial filler text experiment with a
    robust, model-agnostic, and restartable design.
    """
    # --- 1. Load Dependent Data (Centralized Pathing) ---
    # This script uses the definitive path provided by main.py, making it
    # fully compatible with the --restricted flag.
    baseline_results_path = config.BASELINE_RESULTS_PATH
    if not os.path.exists(baseline_results_path):
        logging.error(f"Baseline results file not found at the path provided by main.py: '{baseline_results_path}'")
        return

    logging.info(f"Reading baseline data for model '{config.MODEL_ALIAS}' from '{baseline_results_path}'...")
    all_baseline_trials = []
    with open(baseline_results_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                all_baseline_trials.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping corrupted line {line_num} in {baseline_results_path}")
                continue
    
    # --- 2. Apply Command-Line Filters ---
    # This block correctly enforces the chain limit for dependent experiments.
    if config.NUM_CHAINS_PER_QUESTION > 0:
        logging.info(f"Filtering to include only the first {config.NUM_CHAINS_PER_QUESTION} chains for each question.")
        all_baseline_trials = [
            trial for trial in all_baseline_trials
            if trial['chain_id'] < config.NUM_CHAINS_PER_QUESTION
        ]
    
    # This logic correctly handles the --num-samples flag for quick test runs.
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials: trials_by_question[trial['id']].append(trial)
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # --- 3. Restartability Logic ---
    output_path = config.OUTPUT_PATH
    completed_chains = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        # A chain is complete if it has all 21 percentile steps.
        progress_counts = collections.defaultdict(int)
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    progress_counts[(data['id'], data['chain_id'])] += 1
                except (json.JSONDecodeError, KeyError): continue
        
        for chain_key, count in progress_counts.items():
            if count >= 21: # 21 steps from 0 to 100 inclusive, with step 5
                completed_chains.add(chain_key)

    if completed_chains: logging.info(f"Found {len(completed_chains)} fully completed chains. They will be skipped.")

    # --- 4. Run Experiment ---
    logging.info(f"Running WORD-LEVEL Partial Filler (Random) Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path}")
    
    skipped_trials_count = 0
    with open(output_path, 'a') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if (q_id, chain_id) in completed_chains:
                    continue

                if config.VERBOSE:
                    logging.info(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                # We correctly trust that the 'choices' field is the pre-formatted string.
                choices_formatted = baseline_trial['choices']
                sanitized_cot = baseline_trial['sanitized_cot']
                
                # The main loop iterates through each percentile of corruption.
                for percentile in range(0, 101, 5):
                    if config.VERBOSE:
                        logging.info(f"  - Processing {percentile}% replacement...")
                    
                    # We call our centralized utility to perform the word-level masking.
                    modified_cot = create_word_level_masked_cot(
                        sanitized_cot, percentile, mode='random'
                    )
                    
                    trial_result = run_filler_trial(
                        model, processor, tokenizer, model_utils,
                        baseline_trial['question'], 
                        choices_formatted, 
                        baseline_trial['audio_path'], 
                        modified_cot
                    )

                    # Add all necessary metadata for a complete record.
                    trial_result['id'] = q_id
                    trial_result['chain_id'] = chain_id
                    trial_result['percent_replaced'] = percentile
                    
                    baseline_final_choice = baseline_trial['predicted_choice']
                    trial_result['correct_choice'] = baseline_trial['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == trial_result['correct_choice'])
                    trial_result['corresponding_baseline_predicted_choice'] = baseline_final_choice
                    trial_result['is_consistent_with_baseline'] = (trial_result['predicted_choice'] == baseline_final_choice)
                    
                    # Explicitly order the keys for the final JSON object for readability.
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "chain_id": trial_result['chain_id'],
                        "percent_replaced": trial_result['percent_replaced'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
                        "corresponding_baseline_predicted_choice": trial_result['corresponding_baseline_predicted_choice'],
                        "is_consistent_with_baseline": trial_result['is_consistent_with_baseline'],
                        "final_answer_raw": trial_result['final_answer_raw'],
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "question": trial_result['question'],
                        "choices": trial_result['choices'],
                        "audio_path": trial_result['audio_path']
                    }
                    
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                skipped_trials_count += 1
                logging.exception(f"SKIPPING ENTIRE TRIAL due to unhandled error. ID: {baseline_trial.get('id', 'N/A')}, Chain: {baseline_trial.get('chain_id', 'N/A')}")
                continue

    # --- Final Summary ---
    total_processed_in_this_run = len(samples_to_process) - len(completed_chains)
    logging.info(f"--- WORD-LEVEL Partial Filler (Random) experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total trials in dataset: {len(samples_to_process)}")
    logging.info(f"Trials already complete: {len(completed_chains)}")
    logging.info(f"Trials processed in this run: {total_processed_in_this_run - skipped_trials_count}")
    logging.info(f"Skipped trials due to errors in this run: {skipped_trials_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)