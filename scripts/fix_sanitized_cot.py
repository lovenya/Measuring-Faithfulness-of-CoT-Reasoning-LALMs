#!/usr/bin/env python3
"""
Post-process AF3 baseline JSONL files to fix the double-stripped sanitized_cot.

The original code stripped the last sentence (answer) from the raw output,
then sanitize_cot() stripped ANOTHER sentence via nltk.sent_tokenize.

This script recomputes sanitized_cot correctly: split raw output by sentence
boundaries and remove only the final sentence (the answer).
"""

import json
import re
import os
import sys
import glob

def recompute_sanitized_cot(raw_output: str) -> str:
    """Strip only the last sentence (the answer) from the raw output."""
    if not raw_output:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', raw_output.strip())
    if len(sentences) > 1:
        return " ".join(sentences[:-1])
    return raw_output


def fix_file(filepath: str) -> tuple[int, int]:
    """Fix sanitized_cot in a single JSONL file. Returns (total, changed) counts."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    total = 0
    changed = 0
    fixed_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            fixed_lines.append(line)
            continue

        total += 1
        raw_output = data.get("generated_cot", "")
        old_sanitized = data.get("sanitized_cot", "")
        new_sanitized = recompute_sanitized_cot(raw_output)

        if old_sanitized != new_sanitized:
            changed += 1
            data["sanitized_cot"] = new_sanitized

        fixed_lines.append(json.dumps(data, ensure_ascii=False))

    with open(filepath, 'w') as f:
        for line in fixed_lines:
            f.write(line + "\n")

    return total, changed


if __name__ == "__main__":
    basedir = "results/flamingo_hf/baseline"
    pattern = os.path.join(basedir, "baseline_flamingo_hf_*.jsonl")
    files = sorted(glob.glob(pattern))

    # Exclude .part_N files — only fix the main baseline files
    files = [f for f in files if ".part_" not in f]

    if not files:
        print(f"No baseline files found matching {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} baseline files to fix:")
    for filepath in files:
        print(f"  - {filepath}")
        total, changed = fix_file(filepath)
        print(f"    {total} records, {changed} sanitized_cot fields updated")

    print("\nDone!")
