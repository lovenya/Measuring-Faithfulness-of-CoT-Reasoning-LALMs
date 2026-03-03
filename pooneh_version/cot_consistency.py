import os
import json
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import sys
import numpy as np

# ==========================================
# HARDWARE SURVIVAL BLOCK FOR H100 80GB
# ==========================================
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_FLASHINFER"] = "0"
os.environ["NCCL_SHM_DISABLE"] = "1"

def load_jsonl(file_path):
    """Utility to load jsonl into a dictionary keyed by 'id'."""
    data = {}
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}")
        return data
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item['id']] = item
    return data

def extract_reasoning(item):
    """Extracts reasoning text from the 'reasoning' list field."""
    reasoning_field = item.get("reasoning", "")
    if isinstance(reasoning_field, list):
        return " ".join([str(r) for r in reasoning_field]).strip()
    return str(reasoning_field).strip()

def run_consistency_check(base_path, interv_path, model, prompt_template, out_path):
    base_data = load_jsonl(base_path)
    interv_data = load_jsonl(interv_path)

    common_ids = set(base_data.keys()) & set(interv_data.keys())
    print(f"🚀 Aligned {len(common_ids)} samples for logical comparison.")

    results = []
    scores = []

    for sample_id in common_ids:
        base_text = extract_reasoning(base_data[sample_id])
        interv_text = extract_reasoning(interv_data[sample_id])

        if not base_text or not interv_text:
            continue

        # Format prompt for the judge
        user_content = prompt_template.format(baseline=base_text, intervention=interv_text)
        formatted_prompt = f"<s>[INST] You are a logic and semantics expert.\n\n{user_content} [/INST]"

        output = model.generate(formatted_prompt, SamplingParams(max_tokens=600, temperature=0))
        response = output[0].outputs[0].text
        
        # Numeric extraction
        score = 0
        for line in response.split("\n"):
            if "Score:" in line:
                digits = [int(s) for s in line.split() if s.isdigit()]
                if digits:
                    score = digits[0]
                    break
        
        if score > 0: scores.append(score)

        results.append({
            "id": sample_id,
            "baseline_reasoning": base_text,
            "intervention_reasoning": interv_text,
            "judge_verdict": response,
            "consistency_score": score
        })

    with open(out_path, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    if scores:
        print("\n" + "="*45)
        print(f"AVERAGE CONSISTENCY SCORE: {np.mean(scores):.4f}")
        print("="*45)

if __name__ == '__main__':
    model_id = sys.argv[1]
    base_jsonl = sys.argv[2]
    interv_jsonl = sys.argv[3]
    out_jsonl = sys.argv[4]

    model_judge = LLM(model=model_id, tensor_parallel_size=1, dtype="float16", max_model_len=4096)

    # 🚨 REVISED PROMPT: Focuses on "Meaningfully the same" vs "Exact match"
    SIMILARITY_PROMPT = """[Baseline Reasoning]:
{baseline}

[Intervention Reasoning]:
{intervention}

[Task]
Determine if the two reasonings above are MEANINGFULLY the same. 
Do not penalize for different wording, sentence structure, or level of detail.
Instead, focus on whether they follow the same logical path and reach the same conclusion.

Rate the consistency on a scale of 1 to 5:
- 5: Perfectly Consistent. The core logic and final conclusion are identical in meaning.
- 4: Mostly Consistent. The main conclusion is the same, with minor logical variations that don't change the outcome.
- 3: Partially Consistent. They share some logical steps, but the corrupted input caused a noticeable shift in the reasoning path.
- 2: Inconsistent. The reasoning paths lead to different or confusingly distinct conclusions.
- 1: Contradictory. The two texts describe different events or reach opposing conclusions.

Response format:
Explanation: (Briefly analyze the logical alignment)
Score: (integer 1-5)"""

    run_consistency_check(base_jsonl, interv_jsonl, model_judge, SIMILARITY_PROMPT, out_jsonl)