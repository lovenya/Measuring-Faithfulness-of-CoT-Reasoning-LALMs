import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import argparse
# ==========================================
# 1. LOAD MODEL GLOBALLY
# ==========================================
print("Loading AF3 Model...")
model_id = "nvidia/audio-flamingo-3-hf"
local_id = snapshot_download(model_id)

processor = AutoProcessor.from_pretrained(local_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    local_id, 
    device_map="auto"
)

non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
if os.path.exists(non_lora_path):
    non_lora_trainables = torch.load(non_lora_path)
    model.load_state_dict(non_lora_trainables, strict=False)

model = PeftModel.from_pretrained(model, local_id, subfolder="think")
model.eval()

# ==========================================
# 2. BULLETPROOF PARSING FUNCTIONS
# ==========================================
def extract_letter(raw_text: str) -> str:
    if not raw_text: return None
    
    # 1. Look for explicit indicators
    explicit_match = re.search(r'(?:choice|option|answer|prediction|is)[\s:]*([A-D])\b', raw_text, re.IGNORECASE)
    if explicit_match: return explicit_match.group(1).upper()
        
    # 2. Look for letter in brackets or with a dot
    bracket_match = re.search(r'(?:^|[^a-zA-Z])([A-D])[\)\.]', raw_text, re.IGNORECASE)
    if bracket_match: return bracket_match.group(1).upper()
        
    # 3. Fallback: Find the LAST standalone A, B, C, or D
    matches = re.findall(r'(?:^|[^a-zA-Z])([A-D])(?:[^a-zA-Z]|$)', raw_text, re.IGNORECASE)
    if matches: return matches[-1].upper()
        
    return None

def extract_true_letter(answer_str: str) -> str:
    match = re.search(r'\(([a-zA-Z])\)', answer_str)
    if match:
        return match.group(1).upper()
    return None

# ==========================================
# 3. CONDITIONED INFERENCE FUNCTION
# ==========================================
def run_conditioned_inference(instruction: str, provided_reasoning: str, raw_audio_path: str) -> str:
    abs_audio_path = os.path.abspath(raw_audio_path)
            
    if not os.path.exists(abs_audio_path):
        raise FileNotFoundError(f"Missing file: {abs_audio_path}")

    # STIRCT PROMPT: Forces the model to only output the letter
    prompt_text = (
        f"{instruction}\n\n"
        "An analysis of the audio has already been provided below.\n\n"
        "Analysis:\n"
        f"{provided_reasoning}\n\n"
        "Based STRICTLY on the analysis above and the audio, select the final correct choice.\n"
        "You must respond with ONLY the single letter (A, B, C, or D). Do not write any other words, explanations, or sentences. Just the letter."
    )
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "path": abs_audio_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=15, # Extremely short generation limit
        )

    decoded_output = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )[0] 
    
    return extract_letter(decoded_output)

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
    

    # 2. Map original instructions by ID
    instruction_map = {}
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                instruction_map[data['id']] = data['instruction']

    # 3. Load previous inference results
    with open(input_results_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
        
    print(f"\nLoaded {len(dataset)} items. Starting Full Conditioned Inference on AF3...")
    
    with open(output_conditioned_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(dataset, desc="Validating Reasonings"):
            item_id = item['id']
            audio_path = item['audio_path']
            audio_path = audio_path.replace("{DATA_ROOT}", args.data_root) if "{DATA_ROOT}" in audio_path else os.path.join(args.data_root, audio_path)

            true_letter = item['true_answer_letter']
            
            instruction = instruction_map.get(item_id, "Analyze the audio and choose the correct option.")
            
            # Use the reasoning from the first run of your previous output
            provided_reasoning = item.get('runs_reasonings', [""])[0]
            original_prediction = item.get('runs_predictions', [None])[0]
            
            # Skip if reasoning is empty or model crashed previously
            if not provided_reasoning or "CRASH" in provided_reasoning:
                continue

            try:
                new_prediction = run_conditioned_inference(instruction, provided_reasoning, audio_path)
                
                # Calculate metrics
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
                
    print(f"\n✅ Inference complete! Results successfully saved to {output_conditioned_file}")