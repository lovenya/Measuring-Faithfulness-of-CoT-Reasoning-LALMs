# experiments/adding_mistakes.py

import os
import json
import collections
import nltk

# This is a 'dependent' experiment, as it requires the CoTs from a baseline run.
EXPERIMENT_TYPE = "dependent"

# The few-shot prompt is a core part of the methodology and remains unchanged.
MISTAKE_FEW_SHOT_PROMPT = """<The full few-shot prompt text remains here, unchanged>"""


def generate_mistake(model, processor, model_utils, question: str, choices_formatted: str, original_sentence: str) -> str | None:
    """ 
    Uses the LLM to generate a mistaken version of a sentence.
    """
    prompt = MISTAKE_FEW_SHOT_PROMPT.format(question=question, choices=choices_formatted, original_sentence=original_sentence)
    messages = [{"role": "user", "content": prompt}]
    
    mistaken_sentence = model_utils.run_text_only_inference(
        model, processor, messages, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9
    )
    
    if not mistaken_sentence or not mistaken_sentence.strip():
        return None
    
    return mistaken_sentence.strip()


def continue_reasoning(model, processor, model_utils, audio_path: str, question: str, choices_formatted: str, partial_cot: str) -> str:
    """ Has the model continue generating the CoT from a given starting point. """
    prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": f"Let's think step by step: {partial_cot}"}
    ]
    continuation = model_utils.run_inference(
        model, processor, prompt_messages, audio_path, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9
    )
    return continuation.strip()


def run_final_trial(model, processor, model_utils, question: str, choices_formatted: str, audio_path: str, corrupted_cot: str) -> dict:
    """ Runs the final trial with the fully corrupted CoT to get an answer. """
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices_formatted}"},
        {"role": "assistant", "content": corrupted_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else."}
    ]
    final_answer_text = model_utils.run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False, temperature=0.7, top_p=0.9
    )
    return {
        "predicted_choice": model_utils.parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }


def run(model, processor, model_utils, config):
    """ 
    Orchestrates the 'Adding Mistakes' experiment with robust validation,
    error handling, and restartable design.
    """
    # --- 1. Load Dependent Data (with robust, backward-compatible pathing) ---
    # This block correctly finds the baseline results file, whether it's for our
    # original 'qwen' runs or new runs like 'flamingo' and 'salmonn'.
    if config.MODEL_ALIAS == 'qwen':
        baseline_results_dir = os.path.join(config.RESULTS_DIR, 'baseline')
        baseline_filename = f"baseline_{config.DATASET_NAME}.jsonl"
    else:
        baseline_results_dir = os.path.join(config.RESULTS_DIR, config.MODEL_ALIAS, 'baseline')
        baseline_filename = f"baseline_{config.MODEL_ALIAS}_{config.DATASET_NAME}.jsonl"
    
    baseline_results_path = os.path.join(baseline_results_dir, baseline_filename)

    if not os.path.exists(baseline_results_path):
        print(f"FATAL ERROR: Baseline results file not found for model '{config.MODEL_ALIAS}'. Looked for: '{baseline_results_path}'")
        return

    print(f"Reading baseline data for model '{config.MODEL_ALIAS}' from '{baseline_results_path}'...")
    all_baseline_trials = []
    with open(baseline_results_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # We attempt to load each line as a separate JSON object.
                all_baseline_trials.append(json.loads(line))
            except json.JSONDecodeError as e:
                # If a line is corrupted (e.g., multiple JSON objects mashed together),
                # we print a clear warning and skip that line, allowing the
                # experiment to continue with the valid data.
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING CORRUPTED LINE IN BASELINE RESULTS.")
                print(f"  - File: {baseline_results_path}")
                print(f"  - Line Number: {line_num}")
                print(f"  - Error: {e}")
                print("="*60 + "\n")
                continue
    
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
        print("  - Found existing results file. Checking for completed work...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data['mistake_position'] == data['total_sentences_in_chain']:
                        completed_chains.add((data['id'], data['chain_id']))
                except (json.JSONDecodeError, KeyError): continue
    if completed_chains: print(f"  - Found {len(completed_chains)} fully completed chains. They will be skipped.")

    # --- 3. Run Experiment ---
    print(f"\n--- Running Adding Mistakes Experiment (Model: {config.MODEL_ALIAS.upper()}): Saving to {output_path} ---")
    
    skipped_trials_count = 0
    with open(output_path, 'a') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if (q_id, chain_id) in completed_chains:
                    continue

                if config.VERBOSE:
                    print(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                # We format the choices once here for consistency.
                choices_formatted = model_utils.format_choices_for_prompt(baseline_trial['choices'])
                sanitized_cot = baseline_trial['sanitized_cot']
                sentences = nltk.sent_tokenize(sanitized_cot)
                total_sentences = len(sentences)
                if total_sentences == 0: continue

                for mistake_idx in range(total_sentences):
                    original_sentence = sentences[mistake_idx]

                    if len(original_sentence.split()) < 3:
                        if config.VERBOSE:
                            print(f"  - Skipping sentence {mistake_idx + 1}/{total_sentences} (not meaningful).")
                        continue

                    if config.VERBOSE:
                        print(f"  - Introducing mistake at sentence {mistake_idx + 1}/{total_sentences}...")
                    
                    mistaken_sentence = generate_mistake(model, processor, model_utils, baseline_trial['question'], choices_formatted, original_sentence)
                    
                    if mistaken_sentence is None:
                        if config.VERBOSE:
                            print(f"  - SKIPPING STEP: Model failed to generate a valid mistake for sentence {mistake_idx + 1}.")
                        continue
                    
                    cot_up_to_mistake = " ".join(sentences[:mistake_idx])
                    cot_with_mistake_intro = (cot_up_to_mistake + " " + mistaken_sentence).strip()
                    reasoning_continuation = continue_reasoning(model, processor, model_utils, baseline_trial['audio_path'], baseline_trial['question'], choices_formatted, cot_with_mistake_intro)
                    
                    fully_corrupted_cot = (cot_with_mistake_intro + " " + reasoning_continuation).strip()
                    final_sanitized_corrupted_cot = model_utils.sanitize_cot(fully_corrupted_cot)
                    
                    trial_result = run_final_trial(model, processor, model_utils, baseline_trial['question'], choices_formatted, baseline_trial['audio_path'], final_sanitized_corrupted_cot)

                    baseline_final_choice = baseline_trial['predicted_choice']
                    final_ordered_result = {
                        "id": q_id, "chain_id": chain_id, "mistake_position": mistake_idx + 1,
                        "total_sentences_in_chain": total_sentences, "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": baseline_trial['correct_choice'], "is_correct": (trial_result['predicted_choice'] == baseline_trial['correct_choice']),
                        "corresponding_baseline_predicted_choice": baseline_final_choice,
                        "is_consistent_with_baseline": (trial_result['predicted_choice'] == baseline_final_choice),
                        "final_prompt_messages": trial_result['final_prompt_messages'], "final_answer_raw": trial_result['final_answer_raw']
                    }
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                skipped_trials_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING ENTIRE TRIAL DUE TO AN ERROR.")
                print(f"  - Question ID: {baseline_trial.get('id', 'N/A')}, Chain ID: {baseline_trial.get('chain_id', 'N/A')}")
                print(f"  - Final Error Type: {type(e).__name__}")
                print(f"  - Final Error Details: {e}")
                print("="*60 + "\n")
                continue

    # Final summary
    total_processed_in_this_run = len(samples_to_process) - len(completed_chains)
    print(f"\n--- Adding Mistakes experiment for {config.MODEL_ALIAS.upper()} complete. ---")
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total trials in dataset: {len(samples_to_process)}")
    print(f"Trials already complete: {len(completed_chains)}")
    print(f"Trials processed in this run: {total_processed_in_this_run - skipped_trials_count}")
    print(f"Skipped trials due to errors in this run: {skipped_trials_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)