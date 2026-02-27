import argparse
import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Back to full Conditional Generation class for maximum performance
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- BULLETPROOF CVE BYPASS ---
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni
transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.check_torch_load_is_safe = lambda: None
# ------------------------------

# ==========================================
# 1. LOAD MODEL GLOBALLY
# ==========================================
print("Loading Full Qwen2.5-Omni Model...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0",
    attn_implementation="flash_attention_2", 
)
model.eval()

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# ==========================================
# 2. IMPROVED PARSING
# ==========================================
def simplify(text):
    return re.sub(r'[^\w\s]', '', str(text)).strip().lower()

def parse_model_output(raw_text: str, choices: list = None, use_reasoning: bool = False) -> dict:
    cleaned = raw_text.strip()
    reasoning = ""
    predicted_choice = None
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    if use_reasoning:
        reason_match = re.search(r'<Reason.*?>(.*?)</Reason.*?>', raw_text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()
            
        conclu_match = re.search(r'<Conclu.*?>\s*([A-Za-z])\s*</Conclu.*?>', raw_text, re.IGNORECASE)
        if conclu_match:
            predicted_choice = conclu_match.group(1).upper()

    if not predicted_choice:
        paren_match = re.search(r'\(([A-J])\)', cleaned, re.IGNORECASE)
        if paren_match:
            predicted_choice = paren_match.group(1).upper()
        elif choices:
            for i, opt in enumerate(choices):
                if simplify(cleaned) == simplify(str(opt)):
                    predicted_choice = letters[i]
                    break
        if not predicted_choice:
            punct_match = re.search(r'(?:^|[^a-zA-Z])([A-J])[\.\:]', cleaned, re.IGNORECASE)
            if punct_match:
                predicted_choice = punct_match.group(1).upper()
        if not predicted_choice:
            standalone_matches = list(re.finditer(r'(?:^|[^a-zA-Z])([A-J])(?:[^a-zA-Z]|$)', cleaned, re.IGNORECASE))
            if standalone_matches:
                predicted_choice = standalone_matches[-1].group(1).upper()

    if use_reasoning and not reasoning:
        reasoning = raw_text

    return {"reasoning": reasoning, "predicted_choice": predicted_choice}

# ==========================================
# 3. INFERENCE
# ==========================================
def run_audio_inference(item: dict, data_root: str, use_reasoning: bool) -> dict:
    raw_path = item['audio_path'].replace("{DATA_ROOT}", data_root)
    abs_audio_path = os.path.abspath(raw_path)
            
    if not os.path.exists(abs_audio_path):
        raise FileNotFoundError(f"Missing file: {abs_audio_path}")

    # --- UPDATED PROMPTING ---
    if use_reasoning:
        prompt_text = (
            f"{item['question']} Select one option from the provided choices.\n{item['choices']}. "
            "Please think and reason about the input audio before you respond.\n\n"
            "Template:\n<Reasoning>\n[thinking]\n</Reasoning>\n<Conclusion>\n[Letter]\n</Conclusion>"
        )
        max_tokens = 1024
    else:
        prompt_text = (
            f"{item['question']} Select one option from the provided choices.\n{item['choices']}.\n\n"
            "Template:\n<Conclusion>\n[Letter]\n</Conclusion>"
        )
        max_tokens = 128

    # OFFICIAL SYSTEM PROMPT
    conversation = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": prompt_text}, {"type": "audio", "audio": abs_audio_path}]
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(model.device)
    
    # Cast floating inputs to bfloat16 for Flash Attention compatibility
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)

    with torch.no_grad():
        # Full version returns a tuple (text_ids, audio_ids)
        text_ids, _ = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            use_audio_in_video=False,
            do_sample=False,  # Faster and more consistent for MCQ
            use_cache=True    # Vital for long reasoning speed
        )

    generated_ids = text_ids[:, inputs.input_ids.shape[1]:]
    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    
    parsed = parse_model_output(raw_output, item.get('choices_list', []), use_reasoning)
    return {"raw_output": raw_output, "reasoning": parsed["reasoning"], "predicted_choice": parsed["predicted_choice"]}

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--use_reasoning", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

# 1. Load the original dataset
    with open(args.input, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # 2. CHECK FOR EXISTING PROGRESS
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f_check:
            for line in f_check:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except:
                    continue
        print(f"Resuming: Found {len(processed_ids)} already processed items. Skipping...")

    # 3. Filter the dataset to only include items NOT in processed_ids
    to_process = [item for item in dataset if item['id'] not in processed_ids]

    # 4. Open with 'a' (APPEND mode) instead of 'w'
    with open(args.output, 'a', encoding='utf-8') as f_out:
        for item in tqdm(to_process, desc="Inference"):
            runs_predictions, runs_raw, runs_reasoning, scores = [], [], [], []
            true_letter = item.get("true_letter")

            for run_idx in range(args.num_runs):
                try:
                    result = run_audio_inference(item, args.data_root, args.use_reasoning)
                    pred = result["predicted_choice"]
                    runs_predictions.append(pred)
                    runs_raw.append(result["raw_output"])
                    runs_reasoning.append(result["reasoning"])
                    scores.append(1.0 if (pred and true_letter and pred.upper() == true_letter.upper()) else 0.0)
                except Exception as e:
                    runs_predictions.append(None)
                    runs_raw.append(f"CRASH: {e}")
                    runs_reasoning.append("")
                    scores.append(0.0)

            # Write individual line
            f_out.write(json.dumps({
                "id": item["id"], 
                "true_answer": item.get("answer"), 
                "true_letter": true_letter,
                "predicted_letters": runs_predictions, 
                "reasoning": runs_reasoning,
                "raw_model_outputs": runs_raw, 
                "accuracy": np.mean(scores) if scores else 0.0
            }, ensure_ascii=False) + "\n")
            
            # Flush frequently to ensure data is written to disk if process crashes again
            f_out.flush()