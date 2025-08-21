# experiments/early_answering.py

import os
import json
import collections
import nltk
from core.lalm_utils import run_inference, parse_answer

# This is a 'dependent' experiment because it relies on the CoTs generated
# by a 'baseline' run to perform its analysis.
EXPERIMENT_TYPE = "dependent"

def run_early_answering_trial(model, processor, question: str, choices: str, audio_path: str, truncated_cot: str) -> dict:
    """
    Runs a single trial with a CoT that has been truncated to a specific number of sentences.
    This is the core inference step for the experiment.
    """
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": truncated_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False
    )
    
    # Return a self-documenting dictionary, per our SOP.
    return {
        "predicted_choice": parse_answer(final_answer_text),
        "final_answer_raw": final_answer_text,
        "final_prompt_messages": final_answer_prompt_messages
    }


def run(model, processor, config):
    """
    Orchestrates the full early answering experiment. This version is now optimized
    to reuse pre-computed 'no_reasoning' results for the zero-sentence step.
    """
    # --- 1. Condition-Aware Path Construction for BOTH Dependencies ---
    # This experiment depends on two foundational results: the 'baseline' for the CoTs,
    # and 'no_reasoning' for the 0-sentence step. We must load both from the correct
    # condition-specific directories.

    # --- Path for Baseline Data ---
    if config.CONDITION == 'default':
        baseline_results_dir = os.path.join(config.RESULTS_DIR, 'baseline')
        baseline_filename = f"baseline_{config.DATASET_NAME}.jsonl"
    else:
        condition_specific_results_dir = f"{config.CONDITION}_experiments"
        baseline_results_dir = os.path.join(config.RESULTS_DIR, condition_specific_results_dir, 'baseline')
        baseline_filename = f"baseline_{config.DATASET_NAME}_{config.CONDITION}.jsonl"
    baseline_results_path = os.path.join(baseline_results_dir, baseline_filename)

    # --- Path for No-Reasoning Data ---
    if config.CONDITION == 'default':
        no_reasoning_results_dir = os.path.join(config.RESULTS_DIR, 'no_reasoning')
        no_reasoning_filename = f"no_reasoning_{config.DATASET_NAME}.jsonl"
    else:
        condition_specific_results_dir = f"{config.CONDITION}_experiments"
        no_reasoning_results_dir = os.path.join(config.RESULTS_DIR, condition_specific_results_dir, 'no_reasoning')
        no_reasoning_filename = f"no_reasoning_{config.DATASET_NAME}_{config.CONDITION}.jsonl"
    no_reasoning_results_path = os.path.join(no_reasoning_results_dir, no_reasoning_filename)

    # --- Load Data ---
    try:
        print(f"Reading baseline data for condition '{config.CONDITION}' from '{baseline_results_path}'...")
        all_baseline_trials = [json.loads(line) for line in open(baseline_results_path, 'r')]
        print(f"Reading no-reasoning data for condition '{config.CONDITION}' from '{no_reasoning_results_path}'...")
        all_no_reasoning_trials = [json.loads(line) for line in open(no_reasoning_results_path, 'r')]
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required foundational results file was not found. Details: {e}")
        return
            
    # --- 2. Prepare Data for Efficient Processing ---
    # Create a lookup dictionary for the no-reasoning results for instant access.
    # The key is a tuple of (id, chain_id) for uniqueness.
    no_reasoning_lookup = {(res['id'], res['chain_id']): res for res in all_no_reasoning_trials}

    # Standard logic to handle the --num-samples flag.
    if config.NUM_SAMPLES_TO_RUN > 0:
        trials_by_question = collections.defaultdict(list)
        for trial in all_baseline_trials:
            trials_by_question[trial['id']].append(trial)
        unique_question_ids = list(trials_by_question.keys())[:config.NUM_SAMPLES_TO_RUN]
        samples_to_process = [trial for q_id in unique_question_ids for trial in trials_by_question[q_id]]
    else:
        samples_to_process = all_baseline_trials

    # --- 3. Run the Experiment ---
    output_path = config.OUTPUT_PATH
    print(f"\n--- Running OPTIMIZED Early Answering Experiment ({config.CONDITION} condition): Saving to {output_path} ---")
    print(f"Processing {len(samples_to_process)} total trials.")
    
    skipped_trials_count = 0
    with open(output_path, 'w') as f:
        for i, baseline_trial in enumerate(samples_to_process):
            try:
                q_id, chain_id = baseline_trial['id'], baseline_trial['chain_id']
                if config.VERBOSE:
                    print(f"Processing trial {i+1}/{len(samples_to_process)}: ID {q_id}, Chain {chain_id}")

                sanitized_cot = baseline_trial['sanitized_cot']
                sentences = nltk.sent_tokenize(sanitized_cot)
                total_sentences = len(sentences)
                
                # --- THE CRITICAL OPTIMIZATION ---
                # Step A: Handle the 0-sentence case by reusing pre-computed results.
                lookup_key = (q_id, chain_id)
                nr_result = no_reasoning_lookup.get(lookup_key)

                if nr_result:
                    # Construct the result dictionary for the 0-sentence step.
                    zero_sentence_result = {
                        "id": q_id, "chain_id": chain_id,
                        "num_sentences_provided": 0,
                        "total_sentences_in_chain": total_sentences,
                        "predicted_choice": nr_result['predicted_choice'],
                        "correct_choice": baseline_trial['correct_choice'],
                        "is_correct": (nr_result['predicted_choice'] == baseline_trial['correct_choice']),
                        "corresponding_baseline_predicted_choice": baseline_trial['predicted_choice'],
                        "is_consistent_with_baseline": (nr_result['predicted_choice'] == baseline_trial['predicted_choice']),
                        "final_prompt_messages": nr_result['final_prompt_messages'],
                        "final_answer_raw": nr_result['final_answer_raw']
                    }
                    f.write(json.dumps(zero_sentence_result, ensure_ascii=False) + "\n")
                else:
                    if config.VERBOSE:
                        print(f"  - WARNING: No-reasoning result not found for {lookup_key}. Skipping 0-sentence step.")

                # Step B: Run new inferences ONLY for steps 1 through N.
                for num_sentences_provided in range(1, total_sentences + 1):
                    truncated_cot = " ".join(sentences[:num_sentences_provided])
                    
                    trial_result = run_early_answering_trial(
                        model, processor, baseline_trial['question'], baseline_trial['choices'], baseline_trial['audio_path'], truncated_cot
                    )
                    
                    final_ordered_result = {
                        "id": q_id, "chain_id": chain_id,
                        "num_sentences_provided": num_sentences_provided,
                        "total_sentences_in_chain": total_sentences,
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": baseline_trial['correct_choice'],
                        "is_correct": (trial_result['predicted_choice'] == baseline_trial['correct_choice']),
                        "corresponding_baseline_predicted_choice": baseline_trial['predicted_choice'],
                        "is_consistent_with_baseline": (trial_result['predicted_choice'] == baseline_trial['predicted_choice']),
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "final_answer_raw": trial_result['final_answer_raw']
                    }
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")

            except Exception as e:
                skipped_trials_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING TRIAL DUE TO ERROR.")
                print(f"  - Question ID: {baseline_trial.get('id', 'N/A')}, Chain ID: {baseline_trial.get('chain_id', 'N/A')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # Standard final summary report.
    print("\n--- Early Answering experiment complete. ---")
    print(f"Total trials processed: {len(samples_to_process)}")
    print(f"Skipped trials due to errors: {skipped_trials_count}")
    print(f"Results saved to: {config.OUTPUT_PATH}")