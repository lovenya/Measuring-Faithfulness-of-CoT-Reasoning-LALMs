# experiments/filler_text.py

import os
import json
import collections
from core.lalm_utils import run_inference, parse_answer

EXPERIMENT_TYPE = "dependent"

def run_filler_text_trial(model, processor, question: str, choices: str, audio_path: str, target_token_length: int) -> dict:
    """
    Runs a single trial with filler text of a specific target token length for the LALM.
    """
    filler_text = ""
    if target_token_length > 0:
        filler_unit = "... "
        filler_text = filler_unit * int(target_token_length / 1.5)
        
        # --- CORRECTED METHOD CALL ---
        # We must explicitly use the 'tokenizer' attribute of the processor for text operations.
        while len(processor.tokenizer.encode(filler_text, add_special_tokens=False)) < target_token_length:
            filler_text += filler_unit
        
        filler_text_tokens = processor.tokenizer.encode(filler_text, add_special_tokens=False)[:target_token_length]
        filler_text = processor.tokenizer.decode(filler_text_tokens, skip_special_tokens=True)
        # --- END OF CORRECTION ---

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
        "filler_text_used": filler_text
    }

def run(model, processor, config):
    """
    Orchestrates the full percentile-based filler text experiment for LALMs.
    """
    experiment_name = "filler_text"
    output_dir = os.path.join(config.RESULTS_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{experiment_name}_{config.DATASET_NAME}.jsonl")

    if config.BASELINE_RESULTS_FILE_OVERRIDE:
        baseline_results_path = config.BASELINE_RESULTS_FILE_OVERRIDE
    else:
        baseline_results_path = os.path.join(config.RESULTS_DIR, "baseline", f"baseline_{config.DATASET_NAME}.jsonl")

    if not os.path.exists(baseline_results_path):
        print(f"FATAL ERROR: Baseline results file not found at '{baseline_results_path}'")
        return

    print(f"Reading and grouping baseline data from '{baseline_results_path}'...")
    trials_by_question = collections.defaultdict(list)
    with open(baseline_results_path, 'r') as f:
        for line in f:
            trial = json.loads(line)
            trials_by_question[trial['id']].append(trial)
    
    print(f"\n--- Running Percentile Filler Text Experiment: Saving to {output_path} ---")
    
    skipped_questions_count = 0
    total_questions = len(trials_by_question)

    with open(output_path, 'w') as f:
        for i, (q_id, question_trials) in enumerate(trials_by_question.items()):
            try:
                print(f"Processing question {i+1}/{total_questions}: ID {q_id}")
                
                max_len = 0
                for trial in question_trials:
                    # --- CORRECTED METHOD CALL ---
                    # Explicitly use the 'tokenizer' attribute here as well.
                    cot_len = len(processor.tokenizer.encode(trial['generated_cot']))
                    # --- END OF CORRECTION ---
                    if cot_len > max_len:
                        max_len = cot_len
                
                print(f"  - Max CoT length for this question: {max_len} tokens")
                
                sample_info = question_trials[0]
                
                for percentile in range(0, 101, 5):
                    target_len = int((percentile / 100) * max_len)
                    
                    trial_result = run_filler_text_trial(
                        model, processor, 
                        sample_info['question'], 
                        sample_info['choices'], 
                        sample_info['audio_path'],
                        target_len
                    )

                    trial_result['id'] = q_id
                    trial_result['percentile'] = percentile
                    trial_result['correct_choice'] = sample_info['correct_choice']
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == sample_info['correct_choice'])
                    
                    f.write(json.dumps(trial_result) + "\n")

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
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total unique questions in baseline file: {total_questions}")
    print(f"Successfully processed questions: {total_questions - skipped_questions_count}")
    print(f"Skipped questions due to errors: {skipped_questions_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)