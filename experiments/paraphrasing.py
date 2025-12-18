# experiments/paraphrasing.py

"""
This script conducts the "Paraphrasing" experiment, which tests whether a model's
reasoning is based on the semantic meaning of its CoT or on fragile, specific
phrasing ("magic words").

Methodology:
1. It takes a reasoning chain (CoT) from a baseline run.
2. It progressively paraphrases the CoT, one sentence at a time, using the
   model itself as the paraphrasing engine.
3. At each step, it presents the partially-paraphrased CoT to the model and
   asks for a final answer.
4. By plotting the model's accuracy and consistency against the number of
   paraphrased sentences, we can measure its robustness to changes in phrasing.
   A model that maintains high performance is likely relying on the deeper
   semantic meaning of its reasoning.

External Perturbation Support:
    When config.USE_EXTERNAL_PERTURBATIONS is True, this script loads pre-generated
    paraphrases from a JSONL file (created by scripts/generate_perturbations.py using 
    Mistral Small 3) instead of generating them on-the-fly with the target model.
    This addresses reviewer concerns about in-distribution bias.
"""

import os
import json
import collections
import nltk
import logging

# This is a 'dependent' experiment because it manipulates the CoTs from a 'baseline' run.
EXPERIMENT_TYPE = "dependent"


def load_external_perturbations(perturbation_path: str) -> dict:
    """
    Load pre-generated paraphrases from a JSONL file.
    
    Returns a dictionary keyed by (id, chain_id, num_sentences_paraphrased) -> paraphrased_text.
    The file is created by scripts/generate_perturbations.py using Mistral Small 3.
    """
    perturbations = {}
    if not os.path.exists(perturbation_path):
        logging.error(f"External perturbation file not found: {perturbation_path}")
        return perturbations
    
    with open(perturbation_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # sentence_idx in paraphrase mode = num_sentences_paraphrased
                key = (data['id'], data['chain_id'], data['sentence_idx'])
                perturbations[key] = data['paraphrased_text']
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Skipping malformed perturbation line: {e}")
                continue
    
    logging.info(f"Loaded {len(perturbations)} external paraphrases from {perturbation_path}")
    return perturbations

def get_paraphrased_text(model, processor, tokenizer, model_utils, text_to_paraphrase: str) -> str | None:
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


def run_paraphrasing_trial(model, processor, tokenizer, model_utils, question: str, choices_formatted: str, audio_path: str, modified_cot: str) -> dict:
    """
    Runs a single trial with a (potentially) paraphrased Chain-of-Thought.
    """
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": modified_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False, temperature=0.7, top_p=0.9
    )
    
    # We return a complete dictionary to ensure a robust data flow, mirroring baseline.py.
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
    Orchestrates the full paraphrasing experiment with a robust, model-agnostic,
    and restartable design.
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
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # A chain is considered complete if we have the result for its final paraphrased sentence.
                    if data['num_sentences_paraphrased'] == data['total_sentences_in_chain']:
                        completed_chains.add((data['id'], data['chain_id']))
                except (json.JSONDecodeError, KeyError): continue
    if completed_chains: logging.info(f"Found {len(completed_chains)} fully completed chains. They will be skipped.")

    # --- 3.5. Load External Perturbations (if enabled) ---
    external_perturbations = {}
    use_external = getattr(config, 'USE_EXTERNAL_PERTURBATIONS', False)
    if use_external:
        perturbation_path = getattr(config, 'PERTURBATION_FILE', None)
        if perturbation_path:
            external_perturbations = load_external_perturbations(perturbation_path)
            logging.info(f"Using EXTERNAL paraphrases from Mistral (loaded {len(external_perturbations)} entries)")
        else:
            logging.warning("USE_EXTERNAL_PERTURBATIONS is True but PERTURBATION_FILE not set. Falling back to self-paraphrasing.")
            use_external = False

    # --- 4. Run Experiment ---
    perturbation_source = "EXTERNAL (Mistral)" if use_external else "SELF (target model)"
    logging.info(f"Running Paraphrasing Experiment (Model: {config.MODEL_ALIAS.upper()}, Perturbation: {perturbation_source}): Saving to {output_path}")
    
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
                sentences = nltk.sent_tokenize(sanitized_cot)
                total_sentences = len(sentences)
                if total_sentences == 0: continue

                # We loop from 1 to total_sentences, as the 0% case is just the baseline itself.
                for num_to_paraphrase in range(1, total_sentences + 1):
                    if config.VERBOSE:
                        logging.info(f"  - Paraphrasing {num_to_paraphrase}/{total_sentences} sentences...")
                    
                    part_to_paraphrase = " ".join(sentences[:num_to_paraphrase])
                    remaining_part = " ".join(sentences[num_to_paraphrase:])
                    
                    # Use external paraphrase if available, otherwise generate on-the-fly
                    if use_external:
                        lookup_key = (q_id, chain_id, num_to_paraphrase)
                        paraphrased_part = external_perturbations.get(lookup_key)
                        if paraphrased_part is None or paraphrased_part == "":
                            if config.VERBOSE:
                                logging.info(f"  - SKIPPING STEP: No external paraphrase found for {lookup_key}.")
                            continue
                    else:
                        paraphrased_part = get_paraphrased_text(model, processor, tokenizer, model_utils, part_to_paraphrase)
                        if paraphrased_part is None:
                            if config.VERBOSE:
                                logging.warning(f"  - SKIPPING STEP: Model failed to generate a valid paraphrase for step {num_to_paraphrase}.")
                            continue

                    modified_cot = (paraphrased_part + " " + remaining_part).strip()
                    
                    trial_result = run_paraphrasing_trial(
                        model, processor, tokenizer, model_utils, 
                        baseline_trial['question'], 
                        choices_formatted, 
                        baseline_trial['audio_path'], 
                        modified_cot
                    )
                    
                    # Add the final metadata.
                    trial_result['id'] = q_id
                    trial_result['chain_id'] = chain_id
                    trial_result['num_sentences_paraphrased'] = num_to_paraphrase
                    trial_result['total_sentences_in_chain'] = total_sentences
                    
                    baseline_final_choice = baseline_trial['predicted_choice']
                    trial_result['correct_choice'] = baseline_trial['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == trial_result['correct_choice'])
                    trial_result['corresponding_baseline_predicted_choice'] = baseline_final_choice
                    trial_result['is_consistent_with_baseline'] = (trial_result['predicted_choice'] == baseline_final_choice)
                    
                    # Explicitly order the keys for the final JSON object for readability.
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "chain_id": trial_result['chain_id'],
                        "num_sentences_paraphrased": trial_result['num_sentences_paraphrased'],
                        "total_sentences_in_chain": trial_result['total_sentences_in_chain'],
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
    logging.info(f"--- Paraphrasing experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    logging.info("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    logging.info(f"Total trials in dataset: {len(samples_to_process)}")
    logging.info(f"Trials already complete: {len(completed_chains)}")
    logging.info(f"Trials processed in this run: {total_processed_in_this_run - skipped_trials_count}")
    logging.info(f"Skipped trials due to errors in this run: {skipped_trials_count}")
    logging.info(f"Results saved to: {output_path}")
    logging.info("="*65)