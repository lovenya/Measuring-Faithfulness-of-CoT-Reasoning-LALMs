# experiments/partial_filler_text.py

import os
import json
import collections
import nltk
from core.lalm_utils import run_inference, parse_answer

# This experiment depends on the output of a foundational run.
EXPERIMENT_TYPE = "dependent"

# --- HPC-Safe NLTK Initialization ---
# Ensure NLTK data is available by pointing to the local project folder.
# This is a critical step for offline execution on compute nodes.
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)
else:
    # Provide a clear error if the offline package is missing.
    print(f"FATAL: NLTK data directory not found at '{local_nltk_data_path}'.")
    print("Please ensure you have downloaded the 'punkt' package to this location.")
    # We exit here because the script cannot function without it.
    exit(1)


def run_partial_filler_trial(model, processor, question: str, choices: str, audio_path: str, modified_cot: str) -> dict:
    """Runs a single trial with a CoT that has been partially replaced by filler."""
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": modified_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False
    )
    
    return {
        "predicted_choice": parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }


def run(model, processor, config):
    """
    Orchestrates the partial filler text experiment, adhering to the new SOP.
    """
    # 1. Load dependent data
    baseline_results_path = os.path.join(config.RESULTS_DIR, "baseline", f"baseline_{config.DATASET_NAME}.jsonl")
    if not os.path.exists(baseline_results_path):
        print(f"FATAL ERROR: Baseline results file not found at '{baseline_results_path}'")
        return

    print(f"Reading and grouping baseline data from '{baseline_results_path}'...")
    all_baseline_trials = []
    with open(baseline_results_path, 'r') as f:
        for line in f:
            all_baseline_trials.append(json.loads(line))
            
    # Handle --num-samples by filtering to the first N unique questions
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials:
            trials_by_question[trial['id']].append(trial)
        
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # 2. Run the experiment
    print(f"\n--- Running Partial Filler Text Experiment: Saving to {config.OUTPUT_PATH} ---")
    print(f"Processing {len(samples_to_process)} total trials.")
    
    skipped_trials_count = 0
    with open(config.OUTPUT_PATH, 'w') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id = baseline_trial['id']
                chain_id = baseline_trial['chain_id']
                if config.VERBOSE:
                    print(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                sanitized_cot = baseline_trial['sanitized_cot']
                sentences = nltk.sent_tokenize(sanitized_cot)
                total_sentences = len(sentences)
                
                if total_sentences == 0:
                    if config.VERBOSE:
                        print(f"  - INFO: CoT for trial {q_id}/{chain_id} is empty. Skipping.")
                    continue

                for percentile in range(0, 101, 10):
                    num_sentences_to_replace = int((percentile / 100) * total_sentences)
                    part_to_replace = " ".join(sentences[:num_sentences_to_replace])
                    remaining_part = " ".join(sentences[num_sentences_to_replace:])
                    
                    target_token_length = len(processor.tokenizer.encode(part_to_replace, add_special_tokens=False))
                    filler_text = ""
                    if target_token_length > 0:
                        filler_unit = "... "
                        filler_text = filler_unit * int(target_token_length / 1.5)
                        while len(processor.tokenizer.encode(filler_text, add_special_tokens=False)) < target_token_length:
                            filler_text += filler_unit
                        filler_text_tokens = processor.tokenizer.encode(filler_text, add_special_tokens=False)[:target_token_length]
                        filler_text = processor.tokenizer.decode(filler_text_tokens, skip_special_tokens=True)

                    modified_cot = (filler_text + " " + remaining_part).strip()

                    trial_result = run_partial_filler_trial(
                        model, processor, baseline_trial['question'], baseline_trial['choices'], baseline_trial['audio_path'], modified_cot
                    )

                    trial_result['id'] = q_id
                    trial_result['chain_id'] = chain_id
                    trial_result['percent_replaced'] = percentile
                    trial_result['correct_choice'] = baseline_trial['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == baseline_trial['correct_choice'])
                    
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "chain_id": trial_result['chain_id'],
                        "percent_replaced": trial_result['percent_replaced'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "final_answer_raw": trial_result['final_answer_raw']
                    }
                    f.write(json.dumps(final_ordered_result) + "\n")

            except Exception as e:
                skipped_trials_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING TRIAL DUE TO ERROR.")
                print(f"  - Question ID: {baseline_trial.get('id', 'N/A')}, Chain ID: {baseline_trial.get('chain_id', 'N/A')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # Final summary
    print("\n--- Partial Filler Text experiment complete. ---")
    print(f"Total trials processed: {len(samples_to_process)}")
    print(f"Skipped trials due to errors: {skipped_trials_count}")
    print(f"Results saved to: {config.OUTPUT_PATH}")