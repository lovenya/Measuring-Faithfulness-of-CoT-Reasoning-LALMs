#!/usr/bin/env python3
"""
Pre-flight check: verify that external perturbation files cover all baseline trials.

Run this BEFORE launching parallel experiment jobs to catch Mistral generation gaps early.
Missing coverage means those (id, chain_id, position) entries will be silently skipped
during the experiment run.

For adding_mistakes:
    Checks that every (id, chain_id, sentence_idx) in the baseline has a mistaken_sentence
    in the perturbation file. Only sentence positions with >= 3 words are checked
    (consistent with adding_mistakes.py logic).

For paraphrasing:
    Checks that every (id, chain_id, level) for level in 1..N has a paraphrased_text
    in the perturbation file, where N = number of sentences in that baseline chain.

Usage:
    python scripts/check_perturbation_coverage.py \\
        --model aflamingo_3 --dataset sakura-animal --experiment adding_mistakes

    python scripts/check_perturbation_coverage.py \\
        --model qwen_omni_2.5 --dataset mmar --experiment paraphrasing --restricted
"""

import argparse
import json
import os
import sys

import nltk

RESULTS_DIR = "results"
EXTERNAL_LLM_DIR = os.path.join(RESULTS_DIR, "external_llm_perturbations", "mistral")

# Setup local NLTK data (same convention as main.py)
local_nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if os.path.exists(local_nltk_data_path):
    nltk.data.path.append(local_nltk_data_path)


def get_baseline_path(model: str, dataset: str, restricted: bool) -> str:
    suffix = "-restricted" if restricted else ""
    return os.path.join(RESULTS_DIR, model, "baseline", f"baseline_{model}_{dataset}{suffix}.jsonl")


def get_perturbation_path(model: str, dataset: str, experiment: str, restricted: bool) -> str:
    if experiment == "adding_mistakes":
        name = "mistakes"
    else:
        name = "paraphrased"
    if restricted:
        name += "-restricted"
    return os.path.join(EXTERNAL_LLM_DIR, model, dataset, "raw", f"{name}.jsonl")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: malformed line {i} in {path}: {e}")
    return records


def check_adding_mistakes(baseline: list[dict], perturbations: dict) -> dict:
    """
    For each baseline trial, check all sentence positions are covered.
    perturbations: {(id, chain_id, sentence_idx): mistaken_sentence}
    """
    expected = set()
    for trial in baseline:
        q_id = trial["id"]
        chain_id = trial["chain_id"]
        sentences = nltk.sent_tokenize(trial.get("sanitized_cot", ""))
        for idx, sent in enumerate(sentences):
            if len(sent.split()) >= 3:
                expected.add((q_id, chain_id, idx))

    covered = {k for k in expected if k in perturbations}
    missing = expected - covered
    return {"expected": expected, "covered": covered, "missing": missing}


def check_paraphrasing(baseline: list[dict], perturbations: dict) -> dict:
    """
    For each baseline trial, check all paraphrase levels 1..N are covered.
    perturbations: {(id, chain_id, num_sentences_paraphrased): paraphrased_text}
    """
    expected = set()
    for trial in baseline:
        q_id = trial["id"]
        chain_id = trial["chain_id"]
        sentences = nltk.sent_tokenize(trial.get("sanitized_cot", ""))
        n = len(sentences)
        for level in range(1, n + 1):
            expected.add((q_id, chain_id, level))

    covered = {k for k in expected if k in perturbations}
    missing = expected - covered
    return {"expected": expected, "covered": covered, "missing": missing}


def load_perturbations(path: str, experiment: str) -> dict:
    records = load_jsonl(path)
    result = {}
    for item in records:
        try:
            q_id = item["id"]
            chain_id = item["chain_id"]
            if experiment == "adding_mistakes":
                idx = item["sentence_idx"]
                result[(q_id, chain_id, idx)] = item["mistaken_sentence"]
            else:
                # support both field names for forward compat
                level = item.get("num_sentences_paraphrased", item.get("sentence_idx"))
                result[(q_id, chain_id, level)] = item["paraphrased_text"]
        except KeyError as e:
            print(f"  WARNING: missing field {e} in perturbation record: {item.get('id')}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight check: verify perturbation file covers all baseline trials."
    )
    parser.add_argument("--model", required=True, help="Model alias (canonical).")
    parser.add_argument("--dataset", required=True, help="Dataset alias.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["adding_mistakes", "paraphrasing"],
        help="Experiment type.",
    )
    parser.add_argument(
        "--restricted", action="store_true", help="Use restricted dataset variant."
    )
    parser.add_argument(
        "--results-dir", default="./results", help="Root results directory."
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=20,
        metavar="N",
        help="Print first N missing keys (default: 20, 0 = all).",
    )
    args = parser.parse_args()

    global RESULTS_DIR, EXTERNAL_LLM_DIR
    RESULTS_DIR = args.results_dir
    EXTERNAL_LLM_DIR = os.path.join(RESULTS_DIR, "external_llm_perturbations", "mistral")

    baseline_path = get_baseline_path(args.model, args.dataset, args.restricted)
    perturbation_path = get_perturbation_path(args.model, args.dataset, args.experiment, args.restricted)

    print(f"\n--- Perturbation Coverage Pre-Flight Check ---")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.dataset}{'  [restricted]' if args.restricted else ''}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Baseline:   {baseline_path}")
    print(f"  Perturbs:   {perturbation_path}")

    if not os.path.exists(baseline_path):
        print(f"\nFATAL: Baseline file not found: {baseline_path}")
        return 1

    if not os.path.exists(perturbation_path):
        print(f"\nFATAL: Perturbation file not found: {perturbation_path}")
        return 1

    print("\nLoading files...")
    baseline = load_jsonl(baseline_path)
    perturbations = load_perturbations(perturbation_path, args.experiment)
    print(f"  Baseline trials:  {len(baseline)}")
    print(f"  Perturbation entries: {len(perturbations)}")

    if args.experiment == "adding_mistakes":
        result = check_adding_mistakes(baseline, perturbations)
    else:
        result = check_paraphrasing(baseline, perturbations)

    n_expected = len(result["expected"])
    n_covered = len(result["covered"])
    n_missing = len(result["missing"])
    pct = 100.0 * n_covered / n_expected if n_expected > 0 else 0.0

    print(f"\n--- Coverage Report ---")
    print(f"  Expected positions: {n_expected}")
    print(f"  Covered:            {n_covered}  ({pct:.1f}%)")
    print(f"  Missing:            {n_missing}")

    if n_missing == 0:
        print("\n  RESULT: PASS — full coverage, safe to launch experiment jobs.")
        return 0

    print(f"\n  RESULT: FAIL — {n_missing} positions not covered in perturbation file.")
    print("  These will be silently skipped during the experiment run.")

    limit = args.show_missing if args.show_missing > 0 else n_missing
    missing_sorted = sorted(result["missing"])
    print(f"\n  First {min(limit, n_missing)} missing keys (id, chain_id, position):")
    for key in missing_sorted[:limit]:
        print(f"    {key}")
    if n_missing > limit:
        print(f"    ... and {n_missing - limit} more")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
