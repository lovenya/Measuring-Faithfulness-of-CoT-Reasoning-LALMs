# experiments/filler_text.py

import os
import json
import collections
from core.lalm_utils import run_inference, parse_answer

from core.filler_text_utils import create_filler_for_text

EXPERIMENT_TYPE = "dependent"

def run_filler_text_trial(model, processor, question: str, choices: str, audio_path: str, target_token_length: int) -> dict:
    """
    Runs a single trial with a filler text CoT of a specific token length.
    """
    # --- Use the centralized utility function ---
    # This ensures our filler text generation logic is consistent across all experiments.
    filler_text = create_filler_for_text(processor, " " * target_token_length) # A proxy for token length

    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": filler_text},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]

    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=10, do_sample=False
    )
    
    parsed_choice = parse_answer(final_answer_text)

    return {
        "predicted_choice": parsed_choice,
        "target_token_length": target_token_length,
        "final_prompt_messages": final_answer_prompt_messages,
        "final_answer_raw": final_answer_text
    }


def run(model, processor, config):
    """
    Orchestrates the full percentile-based filler text experiment. This version is
    now fully condition-aware and uses our centralized utility functions.
    """
    output_path = config.OUTPUT_PATH

    # --- 1. Condition-Aware Path Construction for BOTH Dependencies ---
    # This block correctly constructs the paths to the necessary foundational results files,
    # accounting for both the condition-specific directory and the condition-specific filename.
    if config.CONDITION == 'default':
        baseline_results_dir = os.path.join(config.RESULTS_DIR, 'baseline')
        no_reasoning_results_dir = os.path.join(config.RESULTS_DIR, 'no_reasoning')
        baseline_filename = f"baseline_{config.DATASET_NAME}.jsonl"
        no_reasoning_filename = f"no_reasoning_{config.DATASET_NAME}.jsonl"
    else:
        condition_dir = f"{config.CONDITION}_experiments"
        baseline_results_dir = os.path.join(config.RESULTS_DIR, condition_dir, 'baseline')
        no_reasoning_results_dir = os.path.join(config.RESULTS_DIR, condition_dir, 'no_reasoning')
        baseline_filename = f"baseline_{config.DATASET_NAME}_{config.CONDITION}.jsonl"
        no_reasoning_filename = f"no_reasoning_{config.DATASET_NAME}_{config.CONDITION}.jsonl"

    baseline_results_path = os.path.join(baseline_results_dir, baseline_filename)
    no_reasoning_results_path = os.path.join(no_reasoning_results_dir, no_reasoning_filename)

    # --- 2. Load Data ---
    try:
        print(f"Reading baseline data for condition '{config.CONDITION}' from '{baseline_results_path}'...")
        all_baseline_trials = [json.loads(line) for line in open(baseline_results_path, 'r')]
        print(f"Reading no-reasoning data for condition '{config.CONDITION}' from '{no_reasoning_results_path}'...")
        all_no_reasoning_trials = [json.loads(line) for line in open(no_reasoning_results_path, 'r')]
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required foundational results file was not found. Details: {e}")
        return

    # --- 3. Prepare Data for Efficient Processing ---
    trials_by_question = collections.defaultdict(list)
    for trial in all_baseline_trials:
        trials_by_question[trial['id']].append(trial)

    no_reasoning_lookup = {(res['id'], res['chain_id']): res for res in all_no_reasoning_trials}
            
    all_questions_to_process = list(trials_by_question.items())
    if config.NUM_SAMPLES_TO_RUN > 0:
        samples_to_process = all_questions_to_process[:config.NUM_SAMPLES_TO_RUN]
    else:
        samples_to_process = all_questions_to_process

    # --- 4. Run the Experiment ---
    print(f"\n--- Running Optimized Percentile Filler Text Experiment ({config.CONDITION} condition): Saving to {output_path} ---")
    print(f"Processing {len(samples_to_process)} unique questions.")
    
    skipped_questions_count = 0
    with open(output_path, 'w') as f:
        for i, (q_id, question_trials) in enumerate(samples_to_process):
            try:
                if config.VERBOSE:
                    print(f"Processing question {i+1}/{len(samples_to_process)}: ID {q_id}")
                
                # Use the first chain for a given question to get static info
                sample_info = question_trials[0]
                
                # --- Handle the 0% percentile case by reusing no_reasoning data ---
                # We do this for each chain to maintain the same data structure.
                for chain_idx in range(len(question_trials)):
                    lookup_key = (q_id, chain_idx)
                    nr_result = no_reasoning_lookup.get(lookup_key)
                    if nr_result:
                        zero_percentile_prompt = [
                            {"role": "user", "content": f"audio\n\nQuestion: {nr_result['question']}\nChoices:\n{nr_result['choices']}"},
                            {"role": "assistant", "content": ""},
                            {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
                        ]
                        zero_percentile_result = {
                            "id": q_id, "chain_id": chain_idx, "percentile": 0, "target_token_length": 0,
                            "predicted_choice": nr_result['predicted_choice'], "correct_choice": nr_result['correct_choice'],
                            "is_correct": nr_result['is_correct'], "final_prompt_messages": zero_percentile_prompt,
                            "final_answer_raw": nr_result.get('final_answer_raw', '')
                        }
                        f.write(json.dumps(zero_percentile_result, ensure_ascii=False) + "\n")

                # --- Calculate max token length from the SANITIZED CoT ---
                max_len = max(len(processor.tokenizer.encode(t['sanitized_cot'])) for t in question_trials)
                if config.VERBOSE: print(f"  - Max sanitized CoT length for this question: {max_len} tokens")
                if max_len == 0: continue

                # --- Run inferences for 5% to 100% percentiles ---
                for percentile in range(5, 101, 5):
                    target_len = int((percentile / 100) * max_len)
                    
                    trial_result = run_filler_text_trial(
                        model, processor, sample_info['question'], sample_info['choices'], 
                        sample_info['audio_path'], target_len
                    )
                    
                    # Add metadata. Note: we don't save chain_id here as it's a per-question analysis.
                    final_ordered_result = {
                        "id": q_id, "percentile": percentile,
                        "target_token_length": trial_result['target_token_length'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": sample_info['correct_choice'],
                        "is_correct": (trial_result['predicted_choice'] == sample_info['correct_choice']),
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "final_answer_raw": trial_result['final_answer_raw']
                    }
                    f.write(json.dumps(final_ordered_result, ensure_ascii=False) + "\n")

            except Exception as e:
                skipped_questions_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING QUESTION DUE TO ERROR.")
                print(f"  - Question ID: {q_id}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    # --- Final Summary ---
    print("\n--- Percentile filler text experiment complete. ---")
    print(f"Total unique questions processed: {len(samples_to_process)}")
    print(f"Skipped questions due to errors: {skipped_questions_count}")
    print(f"Results saved to: {config.OUTPUT_PATH}")