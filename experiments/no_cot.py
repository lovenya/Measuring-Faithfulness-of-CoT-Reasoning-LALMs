# experiments/no_cot_lalm.py

import os
import json
from core.lalm_utils import run_inference, parse_answer
from data_loader.data_loader import format_choices_for_prompt

EXPERIMENT_TYPE = "foundational"

def run_no_cot_trial(model, processor, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a single "freeflow" trial without an explicit CoT prompt for the LALM model.
    """
    # This is a single-turn prompt that allows the model to generate a brief,
    # spontaneous explanation but guides it to end with a parseable answer.
    direct_answer_prompt = f"audio\n\nQuestion: {question}\nChoices:\n{choices}\n\nWhat is the single, most likely answer? Please ensure your response ends with a single line containing only the letter of the correct choice in parentheses. For example: 'The final answer is: (A)'"

    messages = [
        {"role": "user", "content": direct_answer_prompt}
    ]

    # We use do_sample=True because we are interested in the model's spontaneous
    # reasoning, which can have some variance.
    final_answer_text = run_inference(
        model, processor, messages, audio_path, max_new_tokens=150, do_sample=True
    )
    
    # The parser is robust enough to find the answer even if there's preceding text.
    parsed_choice = parse_answer(final_answer_text)

    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "final_answer_raw": final_answer_text,
        "predicted_choice": parsed_choice,
    }

def run(model, processor, data_samples, config):
    """
    Orchestrates the full "No CoT" LALM experiment with robust error handling.
    """
    experiment_name = "no_cot_lalm"
    output_dir = os.path.join(config.RESULTS_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{experiment_name}_{config.DATASET_NAME}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n--- Running 'No CoT LALM' Experiment: Saving to {output_path} ---")
    
    skipped_samples_count = 0
    with open(output_path, 'w') as f:
        for i, sample in enumerate(data_samples):
            try:
                print(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
                
                choices_formatted = format_choices_for_prompt(sample['choices'])
                
                # We run multiple chains to capture the variance in the model's
                # spontaneous, "freeflow" answers.
                for j in range(config.NUM_CHAINS_PER_QUESTION):
                    print(f"  - Generating chain {j+1}/{config.NUM_CHAINS_PER_QUESTION}...")
                    trial_result = run_no_cot_trial(
                        model, processor, 
                        sample['question'], 
                        choices_formatted, 
                        sample['audio_path']
                    )
                    
                    trial_result['id'] = sample['id']
                    trial_result['chain_id'] = j
                    
                    correct_choice_letter = chr(ord('A') + sample['answer_key'])
                    trial_result['correct_choice'] = correct_choice_letter
                    trial_result['is_correct'] = (trial_result['predicted_choice'] == correct_choice_letter)
                    
                    if 'track' in sample:
                        trial_result['track'] = sample['track']
                    if 'source' in sample:
                        trial_result['source'] = sample['source']
                    if 'hop_type' in sample:
                        trial_result['hop_type'] = sample['hop_type']

                    f.write(json.dumps(trial_result) + "\n")

            except Exception as e:
                skipped_samples_count += 1
                print("\n" + "="*60)
                print(f"WARNING: SKIPPING SAMPLE DUE TO ERROR.")
                print(f"  - Sample ID: {sample.get('id', 'Not Available')}")
                print(f"  - Error Type: {type(e).__name__}")
                print(f"  - Error Details: {e}")
                print("="*60 + "\n")
                continue

    print("\n--- 'No CoT LALM' experiment complete. ---")
    print("\n" + "="*25 + " RUN SUMMARY " + "="*25)
    print(f"Total samples in dataset: {len(data_samples)}")
    print(f"Successfully processed samples: {len(data_samples) - skipped_samples_count}")
    print(f"Skipped samples due to errors: {skipped_samples_count}")
    print(f"Results saved to: {output_path}")
    print("="*65)