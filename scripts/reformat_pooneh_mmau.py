#!/usr/bin/env python3
"""
Reformat Pooneh's MMAU baseline results into our standard format.

Since MMAU uses UUID IDs that are shared between Pooneh's data and ours,
the mapping is trivial — just match by ID.

Note: AF3 MMAU results lack a `reasoning` field (only `raw_model_outputs`).
The sanitized CoT is extracted by stripping the trailing final-answer sentence.
"""

import json
import re
from pathlib import Path


def strip_final_answer(text: str) -> str:
    """Remove the trailing 'The final answer is (X) ...' or 'Therefore, the answer is ...'
    sentence from the raw model output to get the sanitized CoT reasoning."""
    # Match common patterns at the end
    patterns = [
        r'\s*The final answer is \([A-D]\)\s*.*$',
        r'\s*Therefore,?\s*the answer is \([A-D]\)\s*.*$',
        r'\s*The answer is \([A-D]\)\s*.*$',
        r'\s*So,?\s*the answer is \([A-D]\)\s*.*$',
        r'\s*Thus,?\s*the answer is \([A-D]\)\s*.*$',
        r'\s*\*\*\([A-D]\)\s*.*\*\*\s*$',  # **（A) Man**
    ]
    result = text.strip()
    for pat in patterns:
        result = re.sub(pat, '', result, flags=re.IGNORECASE).strip()
    return result


def sanitize_cot(reasoning: str) -> str:
    """Strip XML tags that may wrap model output."""
    reasoning = re.sub(r'<Reasoning>', '', reasoning, flags=re.IGNORECASE)
    reasoning = re.sub(r'</Reasoning>', '', reasoning, flags=re.IGNORECASE)
    return reasoning.strip()


def main():
    base_dir = Path("/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs")
    dataset_path = base_dir / "data/mmau/mmau_test_standardized.jsonl"

    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        print("Please run download_and_normalize_mmau.py first.")
        return

    # Build lookup by UUID (which is now stored as source_id in our data)
    print("Loading our standardized MMAU dataset...")
    our_data = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            # Pooneh's ID corresponds to our source_id
            uuid = d.get('source_id', d['id'])
            our_data[uuid] = d
    print(f"  Loaded {len(our_data)} entries")

    models = {
        "af3": {
            "target": "flamingo_hf",
            "result_file": "baseline_mmau_REAS.jsonl",
            "has_reasoning": False,
        },
        "qwen2.5": {
            "target": "qwen_omni",
            "result_file": "baseline_mmau_REAS.jsonl",
            "has_reasoning": True,
        },
    }

    for source_model, cfg in models.items():
        target_model = cfg["target"]
        pooneh_result = base_dir / f"pooneh_version/result/baseline/{source_model}/mmau/{cfg['result_file']}"

        out_dir = base_dir / f"results/{target_model}/baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"baseline_{target_model}_mmau.jsonl"

        if not pooneh_result.exists():
            print(f"Skipping {source_model} -> {target_model}: {pooneh_result} not found.")
            continue

        print(f"\nProcessing MMAU: {source_model} -> {target_model}...")
        processed = 0
        unmatched = []

        with open(pooneh_result, 'r') as f_in, open(out_file, 'w') as f_out:
            for line in f_in:
                try:
                    poo = json.loads(line)
                except json.JSONDecodeError:
                    continue

                poo_id = poo['id']  # This is the UUID in Pooneh's data
                our_entry = our_data.get(poo_id)

                if our_entry is None:
                    unmatched.append(poo_id)
                    continue

                # Build formatted choices string
                formatted_choices = []
                for idx, c in enumerate(our_entry['choices']):
                    letter = chr(ord('A') + idx)
                    formatted_choices.append(f"({letter}) {c}")
                choices_formatted_str = "\n".join(formatted_choices)

                # Extract reasoning
                if cfg['has_reasoning']:
                    reasoning_list = poo.get('reasoning', [])
                    reasoning = reasoning_list[0] if reasoning_list else ""
                else:
                    # AF3: extract from raw_model_outputs
                    raw_list = poo.get('raw_model_outputs', [])
                    reasoning = raw_list[0] if raw_list else ""

                # Sanitize CoT: strip final answer sentence and XML tags
                sanitized = strip_final_answer(reasoning)
                sanitized = sanitize_cot(sanitized)

                raw_outputs_list = poo.get('raw_model_outputs', [])
                final_answer_raw = raw_outputs_list[0] if raw_outputs_list else ""

                pred_letters = poo.get('predicted_letters', [])
                predicted_choice = pred_letters[0] if pred_letters else ""

                correct_choice = poo.get('true_letter', "")
                is_correct = bool(poo.get('accuracy', 0.0) == 1.0)

                rec = {
                    "id": our_entry["id"],
                    "source_id": our_entry.get("source_id", ""),
                    "chain_id": 0,
                    "predicted_choice": predicted_choice,
                    "correct_choice": correct_choice,
                    "is_correct": is_correct,
                    "final_answer_raw": final_answer_raw,
                    "final_prompt_messages": [],
                    "question": our_entry["question"],
                    "choices": choices_formatted_str,
                    "audio_path": our_entry["audio_path"],
                    "generated_cot": reasoning,
                    "sanitized_cot": sanitized,
                }

                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1

        print(f"  Processed: {processed}  |  Unmatched: {len(unmatched)}")
        if unmatched:
            print("  --- Unmatched IDs (need manual mapping) ---")
            for uid in unmatched[:20]:
                print(f"    {uid}")
            if len(unmatched) > 20:
                print(f"    ... and {len(unmatched) - 20} more")


if __name__ == "__main__":
    main()
