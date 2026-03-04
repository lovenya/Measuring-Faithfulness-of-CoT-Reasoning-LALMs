#!/usr/bin/env python3
"""
Normalizes local MMAU-test-mini Parquet files into our standard format.

This script:
1. Reads one or more HuggingFace-generated Parquet files (e.g., test-00000-of-00002.parquet).
2. Extracts the UUID `id` from the `other_attributes` JSON string.
3. Extracts and decodes the audio `bytes` directly from the Parquet `context` column.
4. Saves each audio file as `data/mmau/audio/mmau_audio_{idx}.wav`.
5. Produces a standardized JSONL: `data/mmau/mmau_test_standardized.jsonl`.
"""

import json
import re
import io
import argparse
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# Output directory (relative to project root).
MMAU_OUTPUT_DIR = Path("data/mmau")


def extract_answer_text(answer_str: str) -> str:
    """Strip the leading letter and parentheses from an answer string."""
    match = re.match(r'^\([a-zA-Z]\)\s*(.+)$', str(answer_str).strip())
    if match:
        return match.group(1).strip()
    return str(answer_str).strip()


def parse_choices(choices_raw) -> list:
    """Parse the choices field. It may be a list of strings or a single string."""
    if isinstance(choices_raw, list):
        import numpy as np
        if isinstance(choices_raw, np.ndarray):
             choices_raw = choices_raw.tolist()
             
        cleaned = []
        for c in choices_raw:
            c_str = str(c).strip()
            m = re.match(r'^\([a-zA-Z]\)\s*(.+)$', c_str)
            if m:
                cleaned.append(m.group(1).strip())
            else:
                cleaned.append(c_str)
        return cleaned
    
    if isinstance(choices_raw, str):
        pattern = r'\([a-zA-Z]\)\s*([^()]+?)(?=\s*\([a-zA-Z]\)|\s*$)'
        matches = re.findall(pattern, choices_raw)
        if matches:
            return [m.strip() for m in matches]
        return [choices_raw.strip()]
    return []


def process_parquet(parquet_path: Path, current_idx: int, audio_dir: Path, jsonl_file, meta_data_list: list) -> int:
    print(f"Reading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Found {len(df)} rows.")

    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {parquet_path.name}"):
        
        # Parse other_attributes
        other_attrs = {}
        other_attrs_raw = row.get("other_attributes", "{}")
        if isinstance(other_attrs_raw, str):
            try:
                other_attrs = json.loads(other_attrs_raw)
            except json.JSONDecodeError:
                pass
        elif isinstance(other_attrs_raw, dict):
             other_attrs = other_attrs_raw
             
        sample_id = other_attrs.get("id", f"mmau_{current_idx}")

        # Save serializable metadata
        serializable = row.to_dict()
        serializable.pop("context", None)
        meta_data_list.append(serializable)

        # Process Audio from bytes
        try:
            audio_bytes = row.get("context.bytes")
            if pd.isna(audio_bytes):
                raise ValueError("context.bytes is empty or NaN")
            
            # Read decoded audio data from memory
            data, samplerate = sf.read(io.BytesIO(audio_bytes))

            audio_filename = f"mmau_audio_{current_idx}.wav"
            audio_filepath = audio_dir / audio_filename
            sf.write(audio_filepath, data, samplerate)

        except Exception as e:
            print(f"\nWarning: Failed to process audio for idx {current_idx} (id={sample_id}). Error: {e}")
            skipped += 1
            current_idx += 1
            continue

        # Parse question, choices, answer
        question = str(row.get("instruction", ""))
        choices = parse_choices(row.get("choices", []))
        answer_raw = str(row.get("answer", ""))
        answer_text = extract_answer_text(answer_raw)

        # Compute answer_key
        try:
            answer_key = choices.index(answer_text)
        except ValueError:
            lower_choices = [c.lower() for c in choices]
            try:
                answer_key = lower_choices.index(answer_text.lower())
            except ValueError:
                print(f"\nWarning: Answer '{answer_text}' not found in choices {choices} for idx {current_idx}. Using 0.")
                answer_key = 0

        # Build standardized record
        normalized = {
            "id": f"mmau_{current_idx}",
            "source_id": sample_id,
            "audio_path": str(audio_filepath),
            "question": question,
            "choices": choices,
            "answer_key": answer_key,
            "answer": answer_text,
            "source": "mmau",
        }

        # Preserve extra metadata
        for extra_key in ["category", "sub-category", "difficulty", "audio_length",
                          "domain", "source_dataset", "task_type"]:
            if extra_key in other_attrs:
                normalized[extra_key] = other_attrs[extra_key]

        jsonl_file.write(json.dumps(normalized, ensure_ascii=False) + '\n')
        current_idx += 1
        
    print(f"  Skipped {skipped} rows in {parquet_path.name}")
    return current_idx


def main():
    parser = argparse.ArgumentParser(description="Normalize local MMAU Parquet files")
    parser.add_argument("parquet_files", nargs="+", type=Path, help="Paths to local .parquet files")
    args = parser.parse_args()

    print("=== MMAU-test-mini Local Normalization ===")

    audio_dir = MMAU_OUTPUT_DIR / "audio"
    MMAU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    jsonl_path = MMAU_OUTPUT_DIR / "mmau_test_standardized.jsonl"
    meta_path = MMAU_OUTPUT_DIR / "MMAU-meta.json"

    print(f"Output directory: {MMAU_OUTPUT_DIR.resolve()}")
    
    current_idx = 0
    original_metadata = []

    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        for p in args.parquet_files:
            if not p.exists():
                print(f"Error: File not found {p}")
                continue
            current_idx = process_parquet(p, current_idx, audio_dir, f_out, original_metadata)

    # Save raw metadata
    with open(meta_path, 'w', encoding='utf-8') as f_meta:
        # Convert numpy types to native python if present
        def default_serializer(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
            
        json.dump(original_metadata, f_meta, indent=2, ensure_ascii=False, default=default_serializer)

    print(f"\n--- MMAU Normalization Complete ---")
    print(f"Processed total rows: {current_idx}")
    print(f"Standardized JSONL: {jsonl_path.resolve()}")
    print(f"Audio directory: {audio_dir.resolve()}")


if __name__ == "__main__":
    main()
