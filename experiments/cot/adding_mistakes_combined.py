# experiments/adding_mistakes_combined.py

"""
Adding Mistakes experiment using pre-combined files.

This is a streamlined version that loads pre-combined data from 
scripts/combine_baseline_with_perturbations.py output files.

Each record in the combined file contains:
- id, chain_id, mistake_position, total_sentences
- prefix_sentences: baseline sentences before the mistake
- mistaken_sentence: the Mistral-generated mistake
- question, choices, audio_path
- correct_choice, baseline_predicted_choice

The experiment:
1. Loads combined data (baseline + Mistral perturbations already merged)
2. For each record, continues reasoning from the mistake point
3. Runs final inference to get the answer
4. Compares with baseline to measure consistency
"""

import os
import json
import logging

# This is a 'dependent' experiment
EXPERIMENT_TYPE = "dependent"


def continue_reasoning(model, processor, tokenizer, model_utils, audio_path: str, question: str, choices_formatted: str, partial_cot: str) -> str:
    """Has the model continue generating the CoT from a given starting point."""
    prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": f"Let's think step by step: {partial_cot}"}
    ]
    continuation = model_utils.run_inference(
        model, processor, prompt_messages, audio_path, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9
    )
    return continuation.strip()


def run_final_trial(model, processor, tokenizer, model_utils, question: str, choices_formatted: str, audio_path: str, corrupted_cot: str) -> dict:
    """Runs the final trial with the fully corrupted CoT to get an answer."""
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": corrupted_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False, temperature=0.7, top_p=0.9
    )
    return {
        "question": question,
        "choices": choices_formatted,
        "audio_path": audio_path,
        "predicted_choice": model_utils.parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }


def run(model, processor, tokenizer, model_utils, config):
    """
    Orchestrates the 'Adding Mistakes' experiment using pre-combined files.
    """
    # --- 1. Load Combined Data ---
    # Auto-detect combined file path if not explicitly provided
    combined_file_path = config.PERTURBATION_FILE
    if not combined_file_path:
        restricted_suffix = "-restricted" if config.RESTRICTED else ""
        combined_file_path = os.path.join(
            "results", "combined",
            f"{config.MODEL_ALIAS}_{config.DATASET_NAME}{restricted_suffix}_adding_mistakes_combined.jsonl"
        )
        logging.info(f"Auto-detected combined file path: {combined_file_path}")
    
    if not os.path.exists(combined_file_path):
        logging.error(f"Combined file not found: {combined_file_path}")
        logging.error("Please run scripts/combine_baseline_with_perturbations.py first.")
        return
    
    logging.info(f"Loading pre-combined data from: {combined_file_path}")
    all_trials = []
    with open(combined_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                all_trials.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping corrupted line {line_num}: {e}")
                continue
    
    logging.info(f"Loaded {len(all_trials)} pre-combined trials")
    
    # --- 2. Apply Command-Line Filters ---
    if config.NUM_CHAINS_PER_QUESTION > 0:
        logging.info(f"Filtering to include only the first {config.NUM_CHAINS_PER_QUESTION} chains for each question.")
        all_trials = [t for t in all_trials if t['chain_id'] < config.NUM_CHAINS_PER_QUESTION]
    
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = {}
        for trial in all_trials:
            q_id = trial['id']
            if q_id not in trials_by_question:
                trials_by_question[q_id] = []
            trials_by_question[q_id].append(trial)
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [t for q_id in unique_question_ids for t in trials_by_question[q_id]]
    else:
        samples_to_process = all_trials
    
    # --- 3. Restartability Logic ---
    output_path = config.OUTPUT_PATH
    completed_keys = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = (data['id'], data['chain_id'], data['mistake_position'])
                    completed_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    if completed_keys:
        logging.info(f"Found {len(completed_keys)} completed trials. They will be skipped.")
    
    # --- 4. Run Experiment ---
    logging.info(f"Running Adding Mistakes Experiment (Model: {config.MODEL_ALIAS.upper()}, Mode: PRE-COMBINED): Saving to {output_path}")
    
    skipped_count = 0
    with open(output_path, 'a') as f:
        for i, trial in enumerate(samples_to_process):
            try:
                q_id = trial['id']
                chain_id = trial['chain_id']
                mistake_position = trial['mistake_position']
                
                # Skip if already completed
                if (q_id, chain_id, mistake_position) in completed_keys:
                    continue
                
                if config.VERBOSE:
                    logging.info(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}, Mistake@{mistake_position}")
                
                # Build partial CoT: prefix sentences + mistaken sentence
                prefix = " ".join(trial['prefix_sentences'])
                mistaken_sentence = trial['mistaken_sentence']
                cot_with_mistake = (prefix + " " + mistaken_sentence).strip()
                
                # Continue reasoning from mistake point
                continuation = continue_reasoning(
                    model, processor, tokenizer, model_utils,
                    trial['audio_path'], trial['question'], trial['choices'],
                    cot_with_mistake
                )
                
                # Build fully corrupted CoT
                fully_corrupted_cot = (cot_with_mistake + " " + continuation).strip()
                final_sanitized_cot = model_utils.sanitize_cot(fully_corrupted_cot)
                
                # Run final inference
                result = run_final_trial(
                    model, processor, tokenizer, model_utils,
                    trial['question'], trial['choices'], trial['audio_path'],
                    final_sanitized_cot
                )
                
                # Add metadata
                result['id'] = q_id
                result['chain_id'] = chain_id
                result['mistake_position'] = mistake_position
                result['total_sentences_in_chain'] = trial['total_sentences']
                result['correct_choice'] = trial['correct_choice']
                result['is_correct'] = (result['predicted_choice'] == trial['correct_choice'])
                result['corresponding_baseline_predicted_choice'] = trial['baseline_predicted_choice']
                result['is_consistent_with_baseline'] = (result['predicted_choice'] == trial['baseline_predicted_choice'])
                
                # Ordered output
                final_result = {
                    "id": result['id'],
                    "chain_id": result['chain_id'],
                    "mistake_position": result['mistake_position'],
                    "total_sentences_in_chain": result['total_sentences_in_chain'],
                    "predicted_choice": result['predicted_choice'],
                    "correct_choice": result['correct_choice'],
                    "is_correct": result['is_correct'],
                    "corresponding_baseline_predicted_choice": result['corresponding_baseline_predicted_choice'],
                    "is_consistent_with_baseline": result['is_consistent_with_baseline'],
                    "final_answer_raw": result['final_answer_raw'],
                    "final_prompt_messages": result['final_prompt_messages'],
                    "question": result['question'],
                    "choices": result['choices'],
                    "audio_path": result['audio_path']
                }
                
                f.write(json.dumps(final_result, ensure_ascii=False) + "\n")
                f.flush()
                
            except Exception as e:
                skipped_count += 1
                logging.exception(f"SKIPPING trial due to error. ID: {trial.get('id', 'N/A')}, Chain: {trial.get('chain_id', 'N/A')}")
                continue
    
    # --- Final Summary ---
    total_processed = len(samples_to_process) - len(completed_keys) - skipped_count
    logging.info(f"--- Adding Mistakes (Combined) experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total trials in dataset: {len(samples_to_process)}")
    logging.info(f"Trials already complete: {len(completed_keys)}")
    logging.info(f"Trials processed in this run: {total_processed}")
    logging.info(f"Skipped due to errors: {skipped_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)
