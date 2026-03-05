import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from peft import PeftModel
from huggingface_hub import snapshot_download
import argparse

# ==========================================
# 1. LOAD MODEL
# ==========================================
model_id = "nvidia/audio-flamingo-3-hf"
local_id = snapshot_download(model_id)
processor = AutoProcessor.from_pretrained(local_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(local_id, device_map="auto")

non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
if os.path.exists(non_lora_path):
    model.load_state_dict(torch.load(non_lora_path), strict=False)
model = PeftModel.from_pretrained(model, local_id, subfolder="think")
model.eval()

# ==========================================
# 2. IMPROVED PARSING
# ==========================================
def simplify(text):
    return re.sub(r'[^\w\s]', '', str(text)).strip().lower()

def parse_model_output(raw_text: str, choices: list = None) -> str:
    cleaned = raw_text.strip()
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # Priority 1: Letter in parentheses (e.g., "(C)")
    paren_match = re.search(r'\(([A-J])\)', cleaned, re.IGNORECASE)
    if paren_match:
        return paren_match.group(1).upper()
    
    # Priority 2: Exact String Matching against choices
    if choices:
        for i, opt in enumerate(choices):
            if simplify(cleaned) == simplify(str(opt)):
                return letters[i]
    
    # Priority 3: Letter followed by period/colon (e.g., "A.")
    punct_match = re.search(r'(?:^|[^a-zA-Z])([A-J])[\.\:]', cleaned, re.IGNORECASE)
    if punct_match:
        return punct_match.group(1).upper()

    # Priority 4: First standalone letter
    standalone_matches = list(re.finditer(r'(?:^|[^a-zA-Z])([A-J])(?:[^a-zA-Z]|$)', cleaned, re.IGNORECASE))
    if standalone_matches:
        return standalone_matches[0].group(1).upper()
        
    return None

# ==========================================
# 3. INFERENCE
# ==========================================
def run_audio_inference(item: dict, data_root: str, use_reasoning: bool) -> dict:
    raw_path = item['audio_path'].replace("{DATA_ROOT}", data_root)
    abs_audio_path = os.path.abspath(raw_path)

    # Dynamic Prompting based on flag
    if use_reasoning:
        prompt_text = (
            f"{item['question']} Select one option from the provided choices.\n{item['choices']}. "
            "Please think and reason about the input audio before you respond."
        )
        max_tokens = 512
    else:
        prompt_text = (
            f"{item['question']} Select one option from the provided choices.\n{item['choices']}. "
        )
        max_tokens = 512

    conversation = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "audio", "path": abs_audio_path},
    ]}]

    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    raw_output = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    reasoning = ""
    prediction_text = raw_output
    
    # If reasoning, isolate the last sentence for parsing
    if use_reasoning:
        sentences = re.split(r'(?<=[.!?])\s+', raw_output.strip())
        if len(sentences) > 1:
            prediction_text = sentences[-1]
            reasoning = " ".join(sentences[:-1])
    
    return {
        "raw_output": raw_output,
        "reasoning": reasoning,
        "predicted_choice": parse_model_output(prediction_text, item.get('choices_list', []))
    }

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--use_reasoning", action="store_true", help="Enable Chain-of-Thought reasoning")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        dataset = [json.loads(line) for line in f]

 # 1. Load the original dataset
    with open(args.input, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # 2. CHECK FOR PREVIOUS PROGRESS
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f_check:
            for line in f_check:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except:
                    # Skip half-written or corrupted lines
                    continue
        print(f"Resuming: {len(processed_ids)} items already completed. Skipping those...")

    # 3. Filter the dataset
    to_process = [item for item in dataset if item['id'] not in processed_ids]

    # 4. Open in APPEND mode ('a')
    # This prevents 'w' from wiping the file on restart
    with open(args.output, 'a', encoding='utf-8') as f_out:
        for item in tqdm(to_process, desc="AF3 Inference"):
            runs_predictions = []
            runs_raw = []
            runs_reasoning = []
            scores = []

            for _ in range(args.num_runs):
                try:
                    result = run_audio_inference(item, args.data_root, args.use_reasoning)
                    pred = result["predicted_choice"]
                    
                    runs_predictions.append(pred)
                    runs_raw.append(result["raw_output"])
                    runs_reasoning.append(result["reasoning"])
                    
                    # Score calculation
                    true_letter = item.get("true_letter")
                    if pred and true_letter and pred.upper() == true_letter.upper():
                        scores.append(1)
                    else:
                        scores.append(0)
                except Exception as e:
                    print(f"\n[ERROR] ID {item['id']} failed: {e}")
                    runs_predictions.append(None)
                    runs_raw.append(f"CRASH: {e}")
                    runs_reasoning.append("")
                    scores.append(0)

            # Write individual result
            f_out.write(json.dumps({
                "id": item["id"],
                "true_answer": item.get("answer"),
                "true_letter": item.get("true_letter"),
                "predicted_letters": runs_predictions,
                "reasoning": runs_reasoning,
                "raw_model_outputs": runs_raw,
                "accuracy": np.mean(scores) if scores else 0.0
            }, ensure_ascii=False) + "\n")
            
            # Flush to disk immediately
            f_out.flush()