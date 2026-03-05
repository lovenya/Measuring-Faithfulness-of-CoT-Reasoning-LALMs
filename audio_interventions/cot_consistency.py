import os
import json
import sys
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
        return data
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data[item['id']] = item
                except json.JSONDecodeError:
                    continue
    return data

def extract_reasoning(item):
    """Extracts reasoning text from the list in the JSON structure."""
    reasoning_field = item.get("reasoning", "")
    if isinstance(reasoning_field, list):
        return " ".join([str(r) for r in reasoning_field]).strip()
    return str(reasoning_field).strip()

def run_consistency_check(base_path, interv_path, model, prompt_template, out_path):
    """Reads JSONL files, aligns by ID, judges reasoning similarity, and saves incrementally."""
    base_data = load_jsonl(base_path)
    interv_data = load_jsonl(interv_path)
    
    # RESUME LOGIC: Check what we already processed
    existing_results = load_jsonl(out_path)
    processed_ids = set(existing_results.keys())
    
    common_ids = set(base_data.keys()) & set(interv_data.keys())
    
    # Filter out already processed IDs
    ids_to_run = [sid for sid in common_ids if sid not in processed_ids]
    
    print(f"🚀 Total aligned samples: {len(common_ids)}")
    print(f"⏭️ Already processed: {len(processed_ids)}")
    print(f"🔥 Samples to run: {len(ids_to_run)}")

    if not ids_to_run:
        print("✅ Everything is already processed. Exiting.")
        return

    # Track scores for final stats, including previously processed ones
    scores_for_stats = [v['similarity_score'] for v in existing_results.values() if v.get('similarity_score', 0) > 0]

    try:
        # Open in append mode 'a' so we don't overwrite existing results
        with open(out_path, 'a') as f:
            for i, sample_id in enumerate(ids_to_run):
                base_reasoning = extract_reasoning(base_data[sample_id])
                interv_reasoning = extract_reasoning(interv_data[sample_id])

                if not base_reasoning or not interv_reasoning:
                    continue

                user_content = prompt_template.format(
                    baseline=base_reasoning, 
                    intervention=interv_reasoning
                )
                formatted_prompt = f"<s>[INST] You are a logic and semantics expert.\n\n{user_content} [/INST]"

                output = model.generate(formatted_prompt, SamplingParams(max_tokens=600, temperature=0))
                response_text = output[0].outputs[0].text
                
                score = 0
                for line in response_text.split("\n"):
                    if "Score:" in line:
                        digits = [int(s) for s in line.split() if s.isdigit()]
                        if digits:
                            score = digits[0]
                            break
                
                if score > 0:
                    scores_for_stats.append(score)

                res_item = {
                    "id": sample_id,
                    "baseline_reasoning": base_reasoning,
                    "intervention_reasoning": interv_reasoning,
                    "judge_explanation": response_text,
                    "similarity_score": score
                }
                
                # Write immediately and flush to disk in case of interruption
                f.write(json.dumps(res_item) + '\n')
                f.flush()

                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(ids_to_run)} new samples...")

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        if scores_for_stats:
            avg_score = np.mean(scores_for_stats)
            print("\n" + "="*45)
            print(f"STATISTICS FOR: {os.path.basename(interv_path)}")
            print(f"Total Samples Scored: {len(scores_for_stats)}")
            print(f"AVERAGE SIMILARITY SCORE: {avg_score:.4f}")
            print("="*45)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python script.py <model_id> <base.jsonl> <interv.jsonl> <output.jsonl>")
        sys.exit(1)

    model_id = sys.argv[1]
    base_jsonl = sys.argv[2]
    interv_jsonl = sys.argv[3]
    out_jsonl = sys.argv[4]

    model_judge = LLM(
        model=model_id,
        tensor_parallel_size=1, 
        dtype="float16",
        max_model_len=4096,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        enforce_eager=True,             
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.9
    )

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
