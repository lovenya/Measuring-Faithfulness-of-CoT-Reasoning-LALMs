#!/usr/bin/env python3
"""
Downloads and normalizes the MMAU-test-mini dataset from Hugging Face.

This script:
1. Loads the `gamma-lab-umd/MMAU-test-mini` dataset from Hugging Face.
2. Extracts the UUID `id` from the `other_attributes` JSON string — this is
   the same ID used in henoop's baseline results, enabling direct mapping.
3. Saves each audio file as `data/mmau/audio/mmau_audio_{idx}.wav`.
4. Produces a standardized JSONL: `data/mmau/mmau_test_standardized.jsonl`.
5. Also saves a raw metadata JSON for traceability.

The schema preserves all metadata from the source (category, sub-category,
difficulty, etc.) alongside the standard fields needed by our framework.
"""

import json
import re
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


# Output directory (relative to project root).
MMAU_OUTPUT_DIR = Path("data/mmau")


def extract_answer_text(answer_str: str) -> str:
    """Strip the leading letter and parentheses from an answer string.
    Example: '(A) Man' -> 'Man'
    """
    match = re.match(r'^\([a-zA-Z]\)\s*(.+)$', answer_str.strip())
    if match:
        return match.group(1).strip()
    return answer_str.strip()


def parse_choices(choices_raw) -> list:
    """Parse the choices field. It may be a list of strings (with letter prefixes)
    or a plain list of answer texts."""
    if isinstance(choices_raw, list):
        cleaned = []
        for c in choices_raw:
            c_str = str(c).strip()
            # Strip "(A) " prefix if present
            m = re.match(r'^\([a-zA-Z]\)\s*(.+)$', c_str)
            if m:
                cleaned.append(m.group(1).strip())
            else:
                cleaned.append(c_str)
        return cleaned
    # If it's a string, try to parse letter-prefixed choices
    if isinstance(choices_raw, str):
        pattern = r'\([a-zA-Z]\)\s*([^()]+?)(?=\s*\([a-zA-Z]\)|\s*$)'
        matches = re.findall(pattern, choices_raw)
        if matches:
            return [m.strip() for m in matches]
        return [choices_raw.strip()]
    return []


def main():
    print("=== MMAU-test-mini Dataset Download & Normalization ===")

    audio_dir = MMAU_OUTPUT_DIR / "audio"
    MMAU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    jsonl_path = MMAU_OUTPUT_DIR / "mmau_test_standardized.jsonl"
    meta_path = MMAU_OUTPUT_DIR / "MMAU-meta.json"

    print(f"Output directory: {MMAU_OUTPUT_DIR.resolve()}")

    # Load the dataset from Hugging Face.
    print("Loading dataset from HuggingFace (gamma-lab-umd/MMAU-test-mini)...")
    try:
        dataset = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test", trust_remote_code=True)
        print(f"Successfully loaded dataset. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"FATAL: Failed to load dataset. Error: {e}")
        return

    skipped = 0
    original_metadata = []

    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        for idx, sample in enumerate(tqdm(dataset, desc="Normalizing MMAU")):

            # --- Parse other_attributes to get UUID id and extra metadata ---
            other_attrs = {}
            other_attrs_raw = sample.get("other_attributes", "{}")
            try:
                other_attrs = json.loads(other_attrs_raw) if isinstance(other_attrs_raw, str) else other_attrs_raw
            except json.JSONDecodeError:
                pass

            sample_id = other_attrs.get("id", f"mmau_{idx}")

            # Save serializable metadata (no audio object).
            serializable = dict(sample)
            serializable.pop("context", None)
            original_metadata.append(serializable)

            # --- Process and save audio ---
            try:
                audio_data = sample["context"]
                audio_array = audio_data["array"]
                sampling_rate = audio_data["sampling_rate"]

                audio_filename = f"mmau_audio_{idx}.wav"
                audio_filepath = audio_dir / audio_filename
                sf.write(audio_filepath, audio_array, sampling_rate)
            except (TypeError, KeyError, AttributeError) as e:
                print(f"\nWarning: Failed to process audio for sample {idx} (id={sample_id}). Error: {e}. Skipping.")
                skipped += 1
                continue

            # --- Parse question, choices, answer ---
            question = sample.get("instruction", "")
            choices = parse_choices(sample.get("choices", []))
            answer_raw = sample.get("answer", "")
            answer_text = extract_answer_text(answer_raw)

            # Compute answer_key
            try:
                answer_key = choices.index(answer_text)
            except ValueError:
                # Try case-insensitive matching
                lower_choices = [c.lower() for c in choices]
                try:
                    answer_key = lower_choices.index(answer_text.lower())
                except ValueError:
                    print(f"\nWarning: Answer '{answer_text}' not found in choices {choices} for sample {idx}. Using 0.")
                    answer_key = 0

            # Build the standardized record, keeping all useful metadata.
            normalized = {
                "id": sample_id,
                "audio_path": str(audio_filepath),
                "question": question,
                "choices": choices,
                "answer_key": answer_key,
                "answer": answer_text,
                "source": "mmau",
            }

            # Preserve extra metadata from other_attributes
            for extra_key in ["category", "sub-category", "difficulty", "audio_length",
                              "domain", "source_dataset", "task_type"]:
                if extra_key in other_attrs:
                    normalized[extra_key] = other_attrs[extra_key]

            f_out.write(json.dumps(normalized, ensure_ascii=False) + '\n')

    # Save raw metadata for traceability.
    with open(meta_path, 'w', encoding='utf-8') as f_meta:
        json.dump(original_metadata, f_meta, indent=2, ensure_ascii=False)

    print(f"\n--- MMAU Normalization Complete ---")
    print(f"Total source samples: {len(dataset)}")
    print(f"Skipped samples: {skipped}")
    print(f"Standardized JSONL: {jsonl_path.resolve()}")
    print(f"Audio directory: {audio_dir.resolve()}")
    print(f"Raw metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
