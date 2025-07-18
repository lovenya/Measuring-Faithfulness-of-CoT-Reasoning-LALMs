# experiments/baseline_lalm.py

import os
import json
from core.lalm_utils import run_inference, parse_answer
from data_loader.data_loader import format_choices_for_prompt


def run_baseline_trial(model, processor, question: str, choices: str, audio_path: str) -> dict:
    """
    Runs a full baseline trial for a single question with audio input.
    """
    # Turn 1: Generate CoT
    cot_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": "Let's think step by step:"}
    ]
    generated_cot = run_inference(
        model, processor, cot_prompt_messages, audio_path, max_new_tokens=768, do_sample=True
    )

    # Turn 2: Elicit Final Answer
    final_answer_prompt_messages = [
        {"role": "user", "content": f"audio\n\nQuestion: {question}\nChoices:\n{choices}"},
        {"role": "assistant", "content": "Let's think step by step: " + generated_cot},
        {"role": "user", "content": "Given the reasoning above, what is the single, most likely answer? Please respond with only the letter of the correct choice in parentheses, and nothing else. For example: (A)"}
    ]
    final_answer_text = run_inference(
        model, processor, final_answer_prompt_messages, audio_path, max_new_tokens=50, do_sample=False
    )
    
    parsed_choice = parse_answer(final_answer_text)

    return {
        "question": question,
        "choices": choices,
        "audio_path": audio_path,
        "generated_cot": generated_cot,
        "final_answer_raw": final_answer_text,
        "predicted_choice": parsed_choice,
    }


def run(model, processor, data_samples, config):
    """Runs the full baseline experiment on a list of data samples and saves the results."""
    all_results = []
    output_filename = f"baseline_lalm_{config.DATASET_NAME}.jsonl"
    output_path = os.path.join(config.RESULTS_DIR, output_filename)
    
    # Ensure results directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"\n--- Running Baseline LALM Experiment: Saving to {output_path} ---")
    with open(output_path, 'w') as f:
        for i, sample in enumerate(data_samples):
            print(f"Processing sample {i+1}/{len(data_samples)}: {sample['id']}")
            
            # Format choices for prompt
            choices_formatted = format_choices_for_prompt(sample['choices'])
            
            for j in range(config.NUM_CHAINS_PER_QUESTION):
                print(f"  - Generating chain {j+1}/{config.NUM_CHAINS_PER_QUESTION}...")
                trial_result = run_baseline_trial(
                    model, processor, 
                    sample['question'], 
                    choices_formatted,
                    sample['audio_path']
                )
                
                # Add metadata for analysis
                trial_result['id'] = sample['id']
                trial_result['chain_id'] = j
                trial_result['correct_choice'] = sample['answer_key']
                trial_result['is_correct'] = (trial_result['predicted_choice'] == sample['answer_key'])
                
                # Add optional metadata if available
                if 'track' in sample:
                    trial_result['track'] = sample['track']
                if 'source' in sample:
                    trial_result['source'] = sample['source']
                if 'hop_type' in sample:
                    trial_result['hop_type'] = sample['hop_type']
                
                # Write each result as a new line in the JSONL file
                f.write(json.dumps(trial_result) + "\n")

    print(f"--- Baseline LALM experiment complete. Results saved to {output_path} ---")