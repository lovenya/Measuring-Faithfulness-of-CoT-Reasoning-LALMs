# data_processing/filter_dependent_results_to_restricted_version.py

"""
Create restricted versions of baseline and dependent experiment results.

This script filters the dataset to only include samples whose baseline CoT
falls within a configurable sentence-length range (default: 1-5 sentences).

HOW IT WORKS:
1. Read the baseline JSONL for a given (model, dataset).
2. Use nltk.sent_tokenize to count sentences in each trial's sanitized_cot.
3. Keep only (id, chain_id) pairs within the sentence range.
4. Write a restricted baseline file.
5. For each dependent experiment, filter rows to only keep those (id, chain_id) pairs.

IMPORTANT: Sentence counting is done ONLY on baseline files. Dependent experiment
results are filtered purely based on whether their (id, chain_id) appears in the
restricted baseline set.

Usage:
    # Single model + dataset:
    python data_processing/filter_dependent_results_to_restricted_version.py \\
        --model flamingo_hf --dataset mmar

    # All datasets for one model:
    python data_processing/filter_dependent_results_to_restricted_version.py \\
        --model flamingo_hf --dataset all

    # Custom range:
    python data_processing/filter_dependent_results_to_restricted_version.py \\
        --model flamingo_hf --dataset all --min-sentences 1 --max-sentences 5
"""

import os
import json
import argparse
import nltk

# Ensure NLTK punkt is available
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_nltk_data_path = os.path.join(PROJECT_ROOT, 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.insert(0, local_nltk_data_path)


# ---- Experiment definitions ----
# Each entry: (experiment_name, has_perturbation_source_variants, has_filler_type_variants)
DEPENDENT_EXPERIMENTS = [
    ("early_answering",              False, False),
    ("adding_mistakes",              True,  False),
    ("paraphrasing",                 True,  False),
    ("random_partial_filler_text",   False, True),
]


def discover_datasets(model: str, results_dir: str) -> list[str]:
    """Find all datasets for a model by scanning the baseline directory."""
    baseline_dir = os.path.join(results_dir, model, 'baseline')
    datasets = set()
    for f in os.listdir(baseline_dir):
        if f.endswith('.jsonl') and '.part_' not in f and '-restricted' not in f:
            # e.g. baseline_flamingo_hf_mmar.jsonl -> mmar
            name = f.replace(f'baseline_{model}_', '').replace('.jsonl', '')
            datasets.add(name)
    return sorted(datasets)


def count_sentences(text: str) -> int:
    """Count sentences using NLTK sent_tokenize."""
    if not text or not text.strip():
        return 0
    return len(nltk.sent_tokenize(text))


def build_restricted_set(baseline_path: str, min_sent: int, max_sent: int, num_chains: int) -> tuple[set, dict]:
    """
    Read baseline, count sentences, return:
      - valid_chains: set of (id, chain_id) tuples within range
      - stats: dict with counts for reporting
    """
    valid_chains = set()
    total = 0
    by_length = {}  # length -> count

    with open(baseline_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            chain_id = data.get('chain_id', 0)
            if num_chains > 0 and chain_id >= num_chains:
                continue

            total += 1
            n_sent = count_sentences(data.get('sanitized_cot', ''))
            by_length[n_sent] = by_length.get(n_sent, 0) + 1

            if min_sent <= n_sent <= max_sent:
                valid_chains.add((data['id'], chain_id))

    return valid_chains, {'total': total, 'kept': len(valid_chains), 'by_length': by_length}


def filter_jsonl(input_path: str, output_path: str, valid_chains: set) -> tuple[int, int]:
    """
    Filter a JSONL file keeping only rows whose (id, chain_id) is in valid_chains.
    Returns (total_read, total_kept).
    """
    total_read = 0
    total_kept = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_read += 1
            if (data['id'], data.get('chain_id', 0)) in valid_chains:
                fout.write(line)
                total_kept += 1

    return total_read, total_kept


def process_one_dataset(model: str, dataset: str, results_dir: str,
                        min_sent: int, max_sent: int, num_chains: int):
    """Process a single (model, dataset) pair: create restricted baseline + filter dependents."""

    print(f"\n{'='*70}")
    print(f"  MODEL: {model.upper()}  |  DATASET: {dataset.upper()}  |  RANGE: {min_sent}-{max_sent} sentences")
    print(f"{'='*70}")

    # ---- Step 1: Build restricted baseline ----
    baseline_path = os.path.join(results_dir, model, 'baseline', f'baseline_{model}_{dataset}.jsonl')
    if not os.path.exists(baseline_path):
        print(f"  SKIP: Baseline not found at {baseline_path}")
        return

    valid_chains, stats = build_restricted_set(baseline_path, min_sent, max_sent, num_chains)
    print(f"\n  Baseline: {stats['total']} trials total, {stats['kept']} within {min_sent}-{max_sent} sentences ({stats['kept']/max(stats['total'],1)*100:.1f}%)")

    # Print length distribution
    print(f"  Sentence distribution (baseline):")
    for length in sorted(stats['by_length'].keys()):
        count = stats['by_length'][length]
        marker = " <-- KEPT" if min_sent <= length <= max_sent else ""
        print(f"    {length:3d} sentences: {count:5d} ({count/stats['total']*100:5.1f}%){marker}")

    # Write restricted baseline
    restricted_baseline_path = baseline_path.replace('.jsonl', '-restricted.jsonl')
    total_read, total_kept = filter_jsonl(baseline_path, restricted_baseline_path, valid_chains)
    print(f"\n  Written restricted baseline: {restricted_baseline_path}")
    print(f"    -> {total_kept}/{total_read} trials kept")

    # ---- Step 2: Filter each dependent experiment ----
    for exp_name, has_pert_source, has_filler_type in DEPENDENT_EXPERIMENTS:
        exp_dir = os.path.join(results_dir, model, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        # Build list of file variants to filter
        variants = []

        # Base file (self / dots)
        base = f"{exp_name}_{model}_{dataset}"
        variants.append(f"{base}.jsonl")

        # Perturbation source variants (e.g. -mistral)
        if has_pert_source:
            variants.append(f"{base}-mistral.jsonl")

        # Filler type variants (e.g. -lorem)
        if has_filler_type:
            variants.append(f"{base}-lorem.jsonl")

        for variant_filename in variants:
            input_path = os.path.join(exp_dir, variant_filename)
            if not os.path.exists(input_path):
                continue

            # Output: insert -restricted before any suffix
            # e.g. adding_mistakes_flamingo_hf_mmar-mistral.jsonl
            #   -> adding_mistakes_flamingo_hf_mmar-restricted-mistral.jsonl
            # e.g. random_partial_filler_text_flamingo_hf_mmar-lorem.jsonl
            #   -> random_partial_filler_text_flamingo_hf_mmar-restricted-lorem.jsonl
            # e.g. early_answering_flamingo_hf_mmar.jsonl
            #   -> early_answering_flamingo_hf_mmar-restricted.jsonl

            # Find where the base dataset name ends to insert -restricted
            base_dataset_end = f"{exp_name}_{model}_{dataset}"
            rest = variant_filename[len(base_dataset_end):]  # e.g. "-mistral.jsonl" or ".jsonl"
            restricted_filename = f"{base_dataset_end}-restricted{rest}"
            output_path = os.path.join(exp_dir, restricted_filename)

            tr, tk = filter_jsonl(input_path, output_path, valid_chains)
            status = "✓" if tk > 0 else "⚠ EMPTY"
            print(f"  {status} {exp_name:40s} {variant_filename:55s} -> {tk}/{tr}")

    print(f"\n  Done with {model}/{dataset}.")


def main():
    parser = argparse.ArgumentParser(
        description="Create restricted baseline + filter dependent experiment results by CoT sentence length.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="Model alias (e.g., 'flamingo_hf'), or 'all'.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset alias (e.g., 'mmar'), or 'all'.")
    parser.add_argument('--results-dir', type=str, default='./results', help="Root results directory.")
    parser.add_argument('--min-sentences', type=int, default=1, help="Minimum sentence count (inclusive). Default: 1.")
    parser.add_argument('--max-sentences', type=int, default=5, help="Maximum sentence count (inclusive). Default: 5.")
    parser.add_argument('--num-chains', type=int, default=1, help="Only process first N chains per question. Default: 1.")
    args = parser.parse_args()

    # Resolve model list
    if args.model == 'all':
        models = sorted([
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d, 'baseline'))
        ])
    else:
        models = [args.model]

    for model in models:
        # Resolve dataset list
        if args.dataset == 'all':
            datasets = discover_datasets(model, args.results_dir)
        else:
            datasets = [args.dataset]

        print(f"\n{'#'*70}")
        print(f"  Processing model: {model.upper()} — {len(datasets)} dataset(s)")
        print(f"{'#'*70}")

        for dataset in datasets:
            process_one_dataset(model, dataset, args.results_dir,
                                args.min_sentences, args.max_sentences, args.num_chains)

    print(f"\n{'='*70}")
    print("  ALL DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
