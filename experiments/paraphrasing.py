# experiments/paraphrasing.py

import os
import json
import collections
import nltk
import logging

# This is a 'dependent' experiment because it manipulates the CoTs from a 'baseline' run.
EXPERIMENT_TYPE = "dependent"

def get_paraphrased_text(model, processor, model_utils, text_to_paraphrase: str) -> str | None:
    """
    Uses the LLM to paraphrase a given piece of text.
    This is a text-only operation and correctly uses the silent audio methodology.
    """
    prompt = f"Please rewrite the following text, conveying exactly the same information but using different wording. Text: \"{text_to_paraphrase}\""
    messages = [{"role": "user", "content": prompt}]
    
    paraphrased_text = model_utils.run_text_only_inference(
        model, processor, messages, max_new_tokens=768, do_sample=True, temperature=0.7, top_p=0.9
    )
    
    # If the model fails to generate a valid paraphrase, we signal a failure.
    if not paraphrased_text or not paraphrased_text.strip():
        return None
        
    return paraphrased_text.strip()


def run_paraphrasing_trial(model, processor, model_utils, question: str, choices_formatted: str, audio_path: str, modified_cot: str) -> dict:
    """
    Runs a single trial with a (potentially) paraphrased Chain-of-Thought.
    
    This function now correctly accepts and returns the question and formatted choices,
    mimicking the robust data flow of the baseline script to prevent data corruption.
    """
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": modified_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False, temperature=0.7, top_p=0.9
    )
    
    # We now return a complete dictionary, just like in baseline.py, to ensure
    # all necessary data is passed back to the main loop.
    return {
        "question": question,
        "choices": choices_formatted, # We return the formatted string
        "audio_path": audio_path,
        "predicted_choice": model_utils.parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }


def run(model, processor, model_utils, config):
    """
    Orchestrates the full paraphrasing experiment with a robust, model-agnostic,
    and restartable design.
    """
    # --- 1. Load Dependent Data (Centralized Pathing) ---
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
    
    # This is the crucial block that correctly enforces the chain limit for
# dependent experiments. It ensures we only process the first N chains for each question,
# making the --num-chains flag behave consistently everywhere.
    if config.NUM_CHAINS_PER_QUESTION > 0: # We assume 10 is the max/default
        logging.info(f"Filtering to include only the first {config.NUM_CHAINS_PER_QUESTION} chains for each question.")
        
        # We create a new list containing only the trials where the chain_id is less than the limit.
        # This correctly handles your scenario (e.g., keeping chains 0, 1 but discarding 3 if the limit is 3).
        filtered_trials = [
            trial for trial in all_baseline_trials
            if trial['chain_id'] < config.NUM_CHAINS_PER_QUESTION
        ]
        
        # We then overwrite the original list with our new, filtered list.
        all_baseline_trials = filtered_trials
    
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials: trials_by_question[trial['id']].append(trial)
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # --- 2. Restartability Logic ---
    output_path = config.OUTPUT_PATH
    completed_chains = set()
    if os.path.exists(output_path):
        logging.info("Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # A chain is considered complete if we have the result for its final paraphrased sentence.
                    if data['num_sentences_paraphrased'] == data['total_sentences_in_chain']:
                        completed_chains.add((data['id'], data['chain_id']))
                except (json.JSONDecodeError, KeyError): continue
    if completed_chains: logging.info(f"Found {len(completed_chains)} fully completed chains. They will be skipped.")

    # --- 3. Run Experiment ---
    logging.info(f"Running Paraphrasing Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path}")
    
    skipped_trials_count = 0
    with open(output_path, 'a') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if (q_id, chain_id) in completed_chains:
                    continue

                if config.VERBOSE:
                    logging.info(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                # We format the choices once here and pass the resulting STRING to all helpers.
                # This was the source of the original data corruption bug.
                choices_formatted = baseline_trial['choices']
                print(f"DEBUG - choices_formatted type: {type(choices_formatted)}")
                print(f"DEBUG - choices_formatted content: {repr(choices_formatted[:200])}")  # First 200 chars
                
                sanitized_cot = baseline_trial['sanitized_cot']
                sentences = nltk.sent_tokenize(sanitized_cot)
                total_sentences = len(sentences)
                if total_sentences == 0: continue

                # We loop from 1 to total_sentences, as the 0% case is just the baseline itself.
                for num_to_paraphrase in range(1, total_sentences + 1):
                    if config.VERBOSE:
                        logging.info(f"  - Paraphrasing {num_to_paraphrase}/{total_sentences} sentences...")
                    
                    part_to_paraphrase = " ".join(sentences[:num_to_paraphrase])
                    remaining_part = " ".join(sentences[num_to_paraphrase:])
                    
                    paraphrased_part = get_paraphrased_text(model, processor, model_utils, part_to_paraphrase)
                    if paraphrased_part is None:
                        if config.VERBOSE:
                            logging.warning(f"  - SKIPPING STEP: Model failed to generate a valid paraphrase for step {num_to_paraphrase}.")
                        continue

                    modified_cot = (paraphrased_part + " " + remaining_part).strip()
                    
                    # The call now correctly passes the formatted choices string.
                    trial_result = run_paraphrasing_trial(
                        model, processor, model_utils, 
                        baseline_trial['question'], 
                        choices_formatted, 
                        baseline_trial['audio_path'], 
                        modified_cot
                    )
                    
                    # The data flow now mirrors baseline.py. We get a complete dictionary back
                    # and simply add the final pieces of metadata.
                    trial_result['id'] = q_id
                    trial_result['chain_id'] = chain_id
                    trial_result['num_sentences_paraphrased'] = num_to_paraphrase
                    trial_result['total_sentences_in_chain'] = total_sentences
                    
                    baseline_final_choice = baseline_trial['predicted_choice']
                    trial_result['correct_choice'] = baseline_trial['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == trial_result['correct_choice'])
                    trial_result['corresponding_baseline_predicted_choice'] = baseline_final_choice
                    trial_result['is_consistent_with_baseline'] = (trial_result['predicted_choice'] == baseline_final_choice)
                    
                    f.write(json.dumps(trial_result, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                skipped_trials_count += 1
                logging.exception(f"SKIPPING ENTIRE TRIAL due to unhandled error. ID: {baseline_trial.get('id', 'N/A')}, Chain: {baseline_trial.get('chain_id', 'N/A')}")
                continue

    # Final summary
    total_processed_in_this_run = len(samples_to_process) - len(completed_chains)
    logging.info(f"--- Paraphrasing experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total trials in dataset: {len(samples_to_process)}")
    logging.info(f"Trials already complete: {len(completed_chains)}")
    logging.info(f"Trials processed in this run: {total_processed_in_this_run - skipped_trials_count}")
    logging.info(f"Skipped trials due to errors in this run: {skipped_trials_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)