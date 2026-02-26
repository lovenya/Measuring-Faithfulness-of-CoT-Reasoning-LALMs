import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import argparse

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ==========================================
# 1. LOAD MODEL GLOBALLY
# ==========================================
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype="auto", 
    device_map="auto",
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model.eval()

# ==========================================
# 2. PARSING FUNCTIONS
# ==========================================
def parse_model_output(raw_text: str) -> dict:
    reasoning = ""
    predicted_choice = None

    # 1. Try forgiving template parsing first
    reason_match = re.search(r'<Reason.*?>(.*?)</Reason.*?>', raw_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()
        
    conclusion_match = re.search(r'<Conclu.*?>\s*([A-Za-z])\s*</Conclu.*?>', raw_text, re.IGNORECASE)
    if conclusion_match:
        predicted_choice = conclusion_match.group(1).upper()
        
    # 2. Fallback if the model completely ignored the tags
    if not predicted_choice:
        cleaned_text = raw_text.strip()
        fallback_match = re.search(r'(?:[^a-zA-Z]|^)([A-D])(?:[^a-zA-Z]*)$', cleaned_text, re.IGNORECASE)
        if fallback_match:
            predicted_choice = fallback_match.group(1).upper()
            if not reasoning:
                reasoning = cleaned_text[:fallback_match.start()].strip()

    return {
        "raw_output": raw_text,
        "reasoning": reasoning if reasoning else raw_text,
        "predicted_choice": predicted_choice
    }

def extract_true_letter(answer_str: str) -> str:
    match = re.search(r'\(([a-zA-Z])\)', answer_str)
    if match:
        return match.group(1).upper()
    return None

# ==========================================
# 3. CONDITIONED INFERENCE FUNCTION
# ==========================================
def run_conditioned_inference(instruction: str, provided_reasoning: str, raw_audio_path: str) -> dict:
    abs_audio_path = os.path.abspath(raw_audio_path)
            
    if not os.path.exists(abs_audio_path):
        raise FileNotFoundError(f"Missing file: {abs_audio_path}")

    # INJECTING THE REASONING INTO YOUR EXACT PROMPT
    prompt_text = (
        f"{instruction}\n\n"
        "You must analyze the audio and provide your answer strictly following the template below. "
        "The analysis has been provided for you; use it to reach the conclusion.\n\n"
        "Template:\n"
        "<Reasoning>\n"
        f"{provided_reasoning}\n"
        "</Reasoning>\n"
        "<Conclusion>\n"
        "[Single letter choice here, e.g., A]\n"
        "</Conclusion>"
    )
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "audio": abs_audio_path},
            ],
        }
    ]

    # YOUR EXACT WORKING INFERENCE LOGIC
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids, generated_audio = model.generate(
            **inputs, 
            max_new_tokens=1024,
            use_audio_in_video=False
        )

    # Slice the text_ids to remove the prompt tokens before decoding
    generated_ids = text_ids[:, inputs.input_ids.shape[1]:]

    decoded_output = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0] 
    
    return parse_model_output(decoded_output)

# ==========================================
# 4. EXECUTION LOOP 
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_file", type=str, required=True, help="Input .jsonl manifest")
    parser.add_argument("--input_results_file", type=str, required=True, help="Output .jsonl results")
    parser.add_argument("--output_conditioned_file", type=str, required=True, help="Output .jsonl results")
    parser.add_argument("--data_root", type=str, default="", help="Root dir for audio files")
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()
    input_results_file = args.input_results_file
    output_conditioned_file = args.output_conditioned_file
    manifest_file = args.manifest_file

    Path(output_conditioned_file).parent.mkdir(parents=True, exist_ok=True)

    instruction_map = {}
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                instruction_map[data['id']] = data['instruction']


    
    Path(output_conditioned_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_results_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loaded {len(dataset)} items. Starting Conditioned Inference...")
    
    with open(output_conditioned_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(dataset, desc="Validating Reasonings"):
            item_id = item['id']
            audio_path = item['audio_path']
            audio_path = audio_path.replace("{DATA_ROOT}", args.data_root) if "{DATA_ROOT}" in audio_path else os.path.join(args.data_root, audio_path)

            true_letter = item['true_answer_letter']
            
            # Retrieve the original instruction
            instruction = instruction_map.get(item_id, "Analyze the audio and choose the correct option.")
            
            # Get the reasoning and prediction from the FIRST run of your previous script
            provided_reasoning = item['runs_reasonings'][0]
            original_prediction = item['runs_predictions'][0]
            
            # Skip if there was no reasoning generated previously
            if not provided_reasoning or "CRASH" in provided_reasoning:
                continue

            try:
                # Run the model WITH the reasoning provided
                result = run_conditioned_inference(instruction, provided_reasoning, audio_path)
                
                new_prediction = result['predicted_choice']
                
                is_correct = 1 if (new_prediction and true_letter and new_prediction == true_letter) else 0
                has_changed = (new_prediction != original_prediction)
                
                final_record = {
                    "id": item_id,
                    "audio_path": audio_path,
                    "true_answer_letter": true_letter,
                    "original_prediction": original_prediction,
                    "conditioned_prediction": new_prediction,
                    "provided_reasoning": provided_reasoning,
                    "is_correct": is_correct,
                    "prediction_changed": has_changed
                }
                
                f_out.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                f_out.flush()
                
            except Exception as e:
                print(f"\n[ERROR] ID: {item_id}: {e}")
                
    print(f"\nConditioned Inference complete! Results saved to {output_conditioned_file}")