# experiments/filler_text.py

import os
import json
import collections
from core.lalm_utils import run_inference, parse_answer


# TODO: Have to use the filler text utils from the core folder


EXPERIMENT_TYPE = "dependent"

def run_filler_text_trial(model, processor, question: str, choices: str, audio_path: str, target_token_length: int) -> dict:
    """
    Runs a single trial with filler text and returns the full prompt used.
    """
    filler_text = ""
    if target_token_length > 0:
        filler_unit = "... "
        filler_text = filler_unit * int(target_token_length / 1.5)
        while len(processor.tokenizer.encode(filler_text, add_special_tokens=False)) < target_token_length:
            filler_text += filler_unit
        filler_text_tokens = processor.tokenizer.encode(filler_text, add_special_tokens=False)[:target_token_length]
        filler_text = processor.tokenizer.decode(filler_text_tokens, skip_special_tokens=True)

    # This is the full prompt object we want to save for transparency.
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
        "final_prompt_messages": final_answer_prompt_messages, # This makes the result self-documenting
        "final_answer_raw": final_answer_text
    }


def run(model, processor, config):
    """
    Orchestrates the full percentile-based filler text experiment.
    This version is highly optimized and logs the full prompt for each trial.
    """
    
    output_path = config.OUTPUT_PATH

    # This experiment now depends on TWO files
    baseline_results_path = os.path.join(config.RESULTS_DIR, "baseline", f"baseline_{config.DATASET_NAME}.jsonl")
    no_reasoning_results_path = os.path.join(config.RESULTS_DIR, "no_reasoning", f"no_reasoning_{config.DATASET_NAME}.jsonl")

    for path in [baseline_results_path, no_reasoning_results_path]:
        if not os.path.exists(path):
            print(f"FATAL ERROR: Dependent results file not found at '{path}'")
            print("Please run both the 'baseline' and 'no_reasoning' experiments first.")
            return

    # 1. Load and process BOTH dependent files
    print(f"Reading baseline data from '{baseline_results_path}'...")
    trials_by_question = collections.defaultdict(list)
    with open(baseline_results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            trials_by_question[data['id']].append(data)

    print(f"Reading no-reasoning data from '{no_reasoning_results_path}'...")
    no_reasoning_results = {}
    with open(no_reasoning_results_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            no_reasoning_results[res['id']] = res
            
            
    all_questions_to_process = list(trials_by_question.items())
    
    # If --num-samples is provided, slice the list of questions.
    if config.NUM_SAMPLES_TO_RUN > 0:
        print(f"\nINFO: --num-samples set to {config.NUM_SAMPLES_TO_RUN}. Processing a subset of questions.")
        samples_to_process = all_questions_to_process[:config.NUM_SAMPLES_TO_RUN]
    else:
        samples_to_process = all_questions_to_process

    # 2. Run the filler text experiment
    print(f"\n--- Running Optimized Percentile Filler Text Experiment: Saving to {output_path} ---")
    print(f"Processing {len(samples_to_process)} unique questions.")
    
    skipped_questions_count = 0
    total_questions = len(trials_by_question)

    with open(output_path, 'w') as f:
        for i, (q_id, question_trials) in enumerate(samples_to_process):
            try:
                if config.VERBOSE:
                    print(f"Processing question {i+1}/{total_questions}: ID {q_id}")
                
                if q_id in no_reasoning_results:
                    nr_result = no_reasoning_results[q_id]
                    
                    # Construct the prompt that *would have been* used for a 0% trial
                    zero_percentile_prompt = [
                        {"role": "user", "content": f"audio\n\nQuestion: {nr_result['question']}\nChoices:\n{nr_result['choices']}"},
                        {"role": "assistant", "content": ""}, # Empty reasoning
                        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
                    ]

                    zero_percentile_result = {
                        "id": q_id,
                        "percentile": 0,
                        "target_token_length": 0,
                        "predicted_choice": nr_result['predicted_choice'],
                        "correct_choice": nr_result['correct_choice'],
                        "is_correct": nr_result['is_correct'],
                        "final_prompt_messages": zero_percentile_prompt,
                        "final_answer_raw": nr_result.get('final_answer_raw', '')
                    }
                    f.write(json.dumps(zero_percentile_result) + "\n")
                else:
                    if config.VERBOSE:
                        print(f"  - WARNING: ID {q_id} not found in no_reasoning results. Skipping 0% point.")

                max_len = max(len(processor.tokenizer.encode(t['generated_cot'])) for t in question_trials)
                
                if config.VERBOSE:
                    print(f"  - Max CoT length for this question: {max_len} tokens")
                

                if max_len == 0:
                    print("  - INFO: All CoTs for this question were empty. Skipping 5-100% trials.")
                    continue

                sample_info = question_trials[0]
                
                for percentile in range(5, 101, 5):
                    target_len = int((percentile / 100) * max_len)
                    
                    # 1. Run the trial to get the results dictionary.
                    trial_result = run_filler_text_trial(
                        model, processor, 
                        sample_info['question'], 
                        sample_info['choices'], 
                        sample_info['audio_path'],
                        target_len
                    )
                    
                    # 2. Add metadata to the dictionary you just received.
                    trial_result['id'] = q_id
                    trial_result['percentile'] = percentile
                    trial_result['correct_choice'] = sample_info['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == sample_info['correct_choice'])
                    
                    # 3. Reorder the dictionary keys for readability before writing.
                    final_ordered_result = {
                        "id": trial_result['id'],
                        "percentile": trial_result['percentile'],
                        "target_token_length": trial_result['target_token_length'],
                        "predicted_choice": trial_result['predicted_choice'],
                        "correct_choice": trial_result['correct_choice'],
                        "is_correct": trial_result['is_correct'],
                        "final_prompt_messages": trial_result['final_prompt_messages'],
                        "final_answer_raw": trial_result['final_answer_raw']
                    }
                    # --- END OF BUG FIX ---
                    
                    f.write(json.dumps(final_ordered_result) + "\n")

            except Exception as e:
                skipped_questions_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING QUESTION DUE TO ERROR.")
                print(f"  - Question ID: {q_id}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    print("\n--- Percentile filler text experiment complete. ---")
    print(f"Total unique questions processed: {len(samples_to_process)}")
    print(f"Skipped questions due to errors: {skipped_questions_count}")
    print(f"Results saved to: {config.OUTPUT_PATH}")
    