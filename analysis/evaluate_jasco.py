#!/usr/bin/env python3
"""
Evaluate JASCO Masking Experiment Results

Reads JASCO experiment output JSONL files and evaluates model responses
via keyword matching against the ground truth keywords from v0.csv.

Reports accuracy breakdown by:
- Audio variant (full / audio_only / speech_only)
- Per-prompt aggregation
- Overall

Usage:
    python analysis/evaluate_jasco.py --results-dir results/qwen/jasco_masking/
    python analysis/evaluate_jasco.py --results-file results/qwen/jasco_masking/jasco_masking_qwen_jasco_full.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


def check_keyword_match(model_output: str, target_keywords: list) -> bool:
    """
    Check if any target keyword appears in the model output (case-insensitive).
    For multi-keyword targets like ["sport", "referee"], returns True if ANY keyword is found.
    """
    output_lower = model_output.lower()
    return any(kw.lower() in output_lower for kw in target_keywords)


def load_results(filepath: str) -> list:
    """Load results from a JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def evaluate_results(results: list) -> dict:
    """
    Evaluate all results and return accuracy metrics.
    
    Returns dict with:
    - overall_accuracy: float
    - per_prompt_accuracy: dict[prompt_index -> accuracy]
    - per_sample: list of per-sample results
    - n_correct: int
    - n_total: int
    """
    per_sample = []
    per_prompt = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    n_correct = 0
    n_total = 0
    
    for entry in results:
        keywords = entry.get('target_keywords', [])
        model_output = entry.get('model_output', '')
        prompt_idx = entry.get('prompt_index', 0)
        
        is_correct = check_keyword_match(model_output, keywords)
        
        per_sample.append({
            'id': entry.get('id'),
            'prompt_index': prompt_idx,
            'chain_id': entry.get('chain_id', 1),
            'variant': entry.get('variant', 'unknown'),
            'is_correct': is_correct,
            'keywords': keywords,
            'model_output_preview': model_output[:100],
        })
        
        per_prompt[prompt_idx]['total'] += 1
        if is_correct:
            per_prompt[prompt_idx]['correct'] += 1
            n_correct += 1
        n_total += 1
    
    overall_accuracy = n_correct / n_total if n_total > 0 else 0.0
    
    per_prompt_accuracy = {}
    for pidx, counts in sorted(per_prompt.items()):
        acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        per_prompt_accuracy[pidx] = {
            'accuracy': acc,
            'correct': counts['correct'],
            'total': counts['total'],
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'n_correct': n_correct,
        'n_total': n_total,
        'per_prompt_accuracy': per_prompt_accuracy,
        'per_sample': per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate JASCO Masking results')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results-dir', type=str,
                       help='Directory containing JASCO result JSONL files (evaluates all)')
    group.add_argument('--results-file', type=str,
                       help='Single JSONL result file to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for detailed per-sample results (JSONL)')
    args = parser.parse_args()
    
    # Collect result files
    if args.results_file:
        files = [args.results_file]
    else:
        result_dir = Path(args.results_dir)
        files = sorted(result_dir.glob('*.jsonl'))
    
    if not files:
        print("No result files found.")
        return
    
    print(f"{'='*70}")
    print(f"JASCO Masking Experiment Evaluation")
    print(f"{'='*70}")
    
    all_variant_results = defaultdict(list)
    
    for filepath in files:
        filepath = str(filepath)
        results = load_results(filepath)
        if not results:
            print(f"  Skipping empty file: {filepath}")
            continue
        
        variant = results[0].get('variant', 'unknown')
        all_variant_results[variant].extend(results)
        
        metrics = evaluate_results(results)
        
        print(f"\n--- {Path(filepath).name} ---")
        print(f"  Variant:  {variant}")
        print(f"  Accuracy: {metrics['n_correct']}/{metrics['n_total']} ({metrics['overall_accuracy']:.1%})")
        
        if len(metrics['per_prompt_accuracy']) > 1:
            print(f"  Per-prompt accuracy:")
            for pidx, pm in metrics['per_prompt_accuracy'].items():
                print(f"    Prompt {pidx}: {pm['correct']}/{pm['total']} ({pm['accuracy']:.1%})")
    
    # Summary across variants
    if len(all_variant_results) > 1:
        print(f"\n{'='*70}")
        print(f"SUMMARY BY VARIANT")
        print(f"{'='*70}")
        print(f"{'Variant':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print(f"{'-'*43}")
        for variant in ['full', 'audio_only', 'speech_only']:
            if variant in all_variant_results:
                metrics = evaluate_results(all_variant_results[variant])
                print(f"{variant:<15} {metrics['n_correct']:>8} {metrics['n_total']:>8} {metrics['overall_accuracy']:>9.1%}")
        print(f"{'-'*43}")
    
    # Save detailed results
    if args.output:
        all_per_sample = []
        for variant, results in all_variant_results.items():
            metrics = evaluate_results(results)
            all_per_sample.extend(metrics['per_sample'])
        
        with open(args.output, 'w') as f:
            for entry in all_per_sample:
                f.write(json.dumps(entry) + '\n')
        print(f"\nDetailed per-sample results saved to: {args.output}")


if __name__ == '__main__':
    main()
