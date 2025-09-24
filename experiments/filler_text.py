# experiments/filler_text.py

"""
This script conducts the "Filler Text" experiment, which is designed to answer
the fundamental question: "Is a Chain-of-Thought helpful because of its semantic
content, or simply because it provides more computational 'thinking time'?"

Methodology:
1. For each question, it finds the SINGLE LONGEST reasoning chain (CoT) from the
   baseline results. This represents the model's "best effort" at reasoning.
2. It then runs a series of trials where the reasoning is replaced by meaningless
   filler text ("...") of varying lengths, from 0% to 100% of the longest CoT's length.
3. The 0% step is optimized by reusing the pre-computed 'no_reasoning' result
   that corresponds to that specific longest chain.
4. By plotting accuracy against the percentage of filler, we can visualize how
   much the model's performance depends on the semantic content versus raw compute.
"""

import os
import json
import collections
import logging
import nltk # We need this for word tokenization
from core.filler_text_utils import create_word_level_masked_cot, run_filler_trial

# This is a 'dependent' experiment because it requires both 'baseline' and
# 'no_reasoning' results to perform its analysis.
EXPERIMENT_TYPE = "dependent"

def run(model, processor, tokenizer, model_utils, config):
    """
    Orchestrates the full percentile-based filler text experiment with a
    robust, model-agnostic, and restartable design.
    """
    # --- 1. Load Dependent Data (Centralized Pathing) ---
    # This script relies on two foundational datasets. It uses the definitive paths
    # provided by main.py, making it fully compatible with the --restricted flag.
    baseline_results_path = config.BASELINE_RESULTS_PATH
    no_reasoning_results_path = config.NO_REASONING_RESULTS_PATH

    for path in [baseline_results_path, no_reasoning_results_path]:
        if not os.path.exists(path):
            logging.error(f"A required foundational results file was not found at: '{path}'")
            return
            
    logging.info(f"Reading baseline data from '{baseline_results_path}'...")
    all_baseline_trials = [json.loads(line) for line in open(baseline_results_path, 'r')]
    
    # --- 2. Apply Command-Line Filters ---
    if config.NUM_CHAINS_PER_QUESTION > 0:
        logging.info(f"Filtering to include only the first {config.NUM_CHAINS_PER_QUESTION} chains for each question.")
        all_baseline_trials = [
            trial for trial in all_baseline_trials
            if trial['chain_id'] < config.NUM_CHAINS_PER_QUESTION
        ]
    
    logging.info(f"Reading no-reasoning data from '{no_reasoning_results_path}'...")
    all_no_reasoning_trials = [json.loads(line) for line in open(no_reasoning_results_path, 'r')]
        
    # --- 3. Prepare Data for Efficient Processing ---
    # We group all baseline chains by their question ID.
    trials_by_question = collections.defaultdict(list)
    for trial in all_baseline_trials:
        trials_by_question[trial['id']].append(trial)

    # We create a lookup dictionary for no-reasoning results for instant access.
    no_reasoning_lookup = {(res['id'], res['chain_id']): res for res in all_no_reasoning_trials}
            
    # This logic correctly handles the --num-samples flag for quick test runs.
    all_questions_to_process = list(trials_by_question.items())
    if config.NUM_SAMPLES_TO_RUN > 0:
        samples_to_process = all_questions_to_process[:config.NUM_SAMPLES_TO_RUN]
    else:
        samples_to_process = all_questions_to_process

    # --- 4. Restartability Logic ---
    output_path = config.OUTPUT_PATH
    completed_questions = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        # A question is considered complete if it has all 21 percentile steps
        # (0% to 100% in 5% increments).
        progress_counts = collections.defaultdict(int)
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    progress_counts[data['id']] += 1
                except (json.JSONDecodeError, KeyError): continue
        
        for q_id, count in progress_counts.items():
            if count >= 21: # 21 steps from 0% to 100%
                completed_questions.add(q_id)

    if completed_questions: logging.info(f"Found {len(completed_questions)} fully completed questions. They will be skipped.")

    # --- 5. Run the Experiment ---
    logging.info(f"Running Percentile Filler Text Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path}")
    
    skipped_questions_count = 0
    with open(output_path, 'a') as f:
        # The main loop iterates through each unique question.
        for i, (q_id, question_trials) in enumerate(samples_to_process):
            try:
                if q_id in completed_questions:
                    continue

                if config.VERBOSE:
                    logging.info(f"Processing question {i+1}/{len(samples_to_process)}: ID {q_id}")
                
                # This experiment is a per-question analysis. We must first identify the
                # single longest reasoning chain, as this represents the model's
                # "best effort" at reasoning for this question.
                longest_chain = max(question_trials, key=lambda t: len(nltk.word_tokenize(t['sanitized_cot'])))
                max_len = len(nltk.word_tokenize(longest_chain['sanitized_cot']))
                
                if max_len == 0:
                    if config.VERBOSE:
                        logging.info(f"  - Skipping question {q_id} as its longest chain is empty.")
                    continue

                # --- Optimization: Reuse No-Reasoning Data for 0% Step ---
                # We now use the no_reasoning result that corresponds ONLY to our
                # identified longest chain.
                lookup_key = (q_id, longest_chain['chain_id'])
                nr_result = no_reasoning_lookup.get(lookup_key)
                if nr_result:
                    # We add the 'percentile' key for consistency in our data format.
                    zero_percentile_result = {
                        "id": q_id, "chain_id": longest_chain['chain_id'], "percentile": 0,
                        "predicted_choice": nr_result['predicted_choice'], "correct_choice": nr_result['correct_choice'],
                        "is_correct": nr_result['is_correct'], "final_prompt_messages": nr_result['final_prompt_messages'],
                        "final_answer_raw": nr_result.get('final_answer_raw', ''),
                        "question": longest_chain['question'], "choices": longest_chain['choices'],
                        "audio_path": longest_chain['audio_path']
                    }
                    f.write(json.dumps(zero_percentile_result, ensure_ascii=False) + "\n")
                    f.flush()

                # --- Run Inferences for 5% to 100% Percentiles ---
                # This loop constitutes the main "compute-simulation-response" part of the experiment.
                for percentile in range(5, 101, 5):
                    # The filler text is now based on the length of the single longest chain.
                    modified_cot = create_word_level_masked_cot(" " * max_len, percentile, mode='start')
                    
                    trial_result = run_filler_trial(
                        model, processor, tokenizer, model_utils, 
                        longest_chain['question'], 
                        longest_chain['choices'], 
                        longest_chain['audio_path'], 
                        modified_cot
                    )
                    
                    
                    trial_result['id'] = q_id
                    trial_result['chain_id'] = longest_chain['chain_id']
                    trial_result['percentile'] = percentile
                    trial_result['correct_choice'] = longest_chain['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == longest_chain['correct_choice'])
                    
                    
                    # We save a single result per percentile, but now we also record the
                    # chain_id of the longest chain for full traceability.
                    # And also - explicitly order the keys for the final JSON object for readability.
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "chain_id": trial_result['chain_id'],
                        "percentile": trial_result['percentile'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
                        "final_answer_raw": trial_result['final_answer_raw'],
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "question": trial_result['question'],
                        "choices": trial_result['choices'],
                        "audio_path": trial_result['audio_path']
                    }
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                skipped_questions_count += 1
                logging.exception(f"SKIPPING QUESTION due to unhandled error. ID: {q_id}")
                continue

    # --- Final Summary ---
    total_processed_in_this_run = len(samples_to_process) - len(completed_questions)
    logging.info(f"--- Percentile Filler Text experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total unique questions in dataset: {len(samples_to_process)}")
    logging.info(f"Questions already complete: {len(completed_questions)}")
    logging.info(f"Questions processed in this run: {total_processed_in_this_run - skipped_questions_count}")
    logging.info(f"Skipped questions due to errors in this run: {skipped_questions_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)