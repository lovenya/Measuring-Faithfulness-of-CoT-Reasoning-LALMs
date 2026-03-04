# data_processing/diagnose_no_modification_accuracy.py

"""
Diagnostic script to calculate accuracies at the "no modification" control points
across three experiments:

1. Random Partial Filler Text at 0% replacement
2. Paraphrasing at 0% paraphrased
3. Early Answering at 100% sentences provided

These all represent the unmodified CoT, so their accuracies should theoretically
match the baseline. Any deviation reveals noise from the experimental pipeline.

Usage:
    python data_processing/diagnose_no_modification_accuracy.py --model flamingo_hf
    python data_processing/diagnose_no_modification_accuracy.py --model flamingo_hf --restricted
    python data_processing/diagnose_no_modification_accuracy.py --model all
"""

import os
import sys
import json
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
from utils import load_results, discover_datasets


def compute_accuracy(df: pd.DataFrame) -> dict:
    """Compute accuracy from a DataFrame with is_correct column."""
    if df.empty:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}
    total = len(df)
    correct = int(df['is_correct'].sum())
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total * 100
    }


def analyze_model(model: str, results_dir: str, restricted: bool, 
                   perturbation_source: str, filler_type: str):
    """Run the full no-modification accuracy analysis for one model."""
    
    datasets = discover_datasets(model, results_dir)
    restricted_label = " [RESTRICTED]" if restricted else ""
    
    print(f"\n{'#'*70}")
    print(f"  MODEL: {model.upper()}{restricted_label}")
    print(f"  Perturbation Source: {perturbation_source.upper()}, Filler Type: {filler_type.upper()}")
    print(f"{'#'*70}")
    
    # Collect all results into a table
    rows = []
    
    for dataset in datasets:
        row = {'dataset': dataset}
        
        # --- Baseline Accuracy ---
        try:
            baseline_df = load_results(model, results_dir, 'baseline', dataset, restricted=restricted)
            bl = compute_accuracy(baseline_df)
            row['bl_total'] = bl['total']
            row['bl_acc'] = bl['accuracy']
        except FileNotFoundError:
            row['bl_total'] = 0
            row['bl_acc'] = float('nan')
        
        # --- Random Partial Filler at 0% ---
        try:
            rpf_df = load_results(model, results_dir, 'random_partial_filler_text', dataset,
                                   filler_type=filler_type, restricted=restricted)
            rpf_0 = rpf_df[rpf_df['percent_replaced'] == 0]
            rpf_stats = compute_accuracy(rpf_0)
            row['rpf_total'] = rpf_stats['total']
            row['rpf_acc'] = rpf_stats['accuracy']
        except FileNotFoundError:
            row['rpf_total'] = 0
            row['rpf_acc'] = float('nan')
        
        # --- Paraphrasing at 0% ---
        try:
            par_df = load_results(model, results_dir, 'paraphrasing', dataset,
                                   perturbation_source=perturbation_source, restricted=restricted)
            par_0 = par_df[par_df['num_sentences_paraphrased'] == 0]
            par_stats = compute_accuracy(par_0)
            row['par_total'] = par_stats['total']
            row['par_acc'] = par_stats['accuracy']
        except FileNotFoundError:
            row['par_total'] = 0
            row['par_acc'] = float('nan')
        
        # --- Early Answering at 100% ---
        try:
            ea_df = load_results(model, results_dir, 'early_answering', dataset, restricted=restricted)
            ea_df = ea_df[ea_df['total_sentences_in_chain'] > 0].copy()
            ea_100 = ea_df[ea_df['num_sentences_provided'] == ea_df['total_sentences_in_chain']]
            ea_stats = compute_accuracy(ea_100)
            row['ea_total'] = ea_stats['total']
            row['ea_acc'] = ea_stats['accuracy']
        except FileNotFoundError:
            row['ea_total'] = 0
            row['ea_acc'] = float('nan')
        
        rows.append(row)
    
    # --- Print Table ---
    print(f"\n  {'Dataset':<20} {'Baseline':>10} {'RPF@0%':>10} {'PAR@0%':>10} {'EA@100%':>10}")
    print(f"  {'':20s} {'Acc (n)':>10} {'Acc (n)':>10} {'Acc (n)':>10} {'Acc (n)':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for r in rows:
        def fmt(acc_key, total_key):
            if r.get(total_key, 0) == 0:
                return "   N/A   "
            return f"{r[acc_key]:5.1f}({r[total_key]:>4d})"
        
        print(f"  {r['dataset']:<20} {fmt('bl_acc', 'bl_total'):>10} {fmt('rpf_acc', 'rpf_total'):>10} {fmt('par_acc', 'par_total'):>10} {fmt('ea_acc', 'ea_total'):>10}")
    
    # --- Aggregated row ---
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    def agg(acc_key, total_key):
        total = sum(r.get(total_key, 0) for r in rows)
        if total == 0:
            return "   N/A   ", 0.0
        # Weighted average
        weighted_sum = sum(r.get(acc_key, 0) * r.get(total_key, 0) for r in rows if r.get(total_key, 0) > 0)
        avg = weighted_sum / total
        return f"{avg:5.1f}({total:>4d})", avg
    
    bl_str, _ = agg('bl_acc', 'bl_total')
    rpf_str, _ = agg('rpf_acc', 'rpf_total')
    par_str, _ = agg('par_acc', 'par_total')
    ea_str, _ = agg('ea_acc', 'ea_total')
    print(f"  {'AVERAGE (weighted)':<20} {bl_str:>10} {rpf_str:>10} {par_str:>10} {ea_str:>10}")
    
    # --- Deviation analysis ---
    print(f"\n  Deviation from Baseline:")
    for r in rows:
        if r.get('bl_total', 0) == 0:
            continue
        bl = r['bl_acc']
        deviations = []
        for label, acc_key, total_key in [('RPF@0%', 'rpf_acc', 'rpf_total'), 
                                           ('PAR@0%', 'par_acc', 'par_total'),
                                           ('EA@100%', 'ea_acc', 'ea_total')]:
            if r.get(total_key, 0) > 0:
                diff = r[acc_key] - bl
                sign = "+" if diff >= 0 else ""
                deviations.append(f"{label}: {sign}{diff:.1f}pp")
        if deviations:
            print(f"    {r['dataset']:<20} {', '.join(deviations)}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy at no-modification control points across experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="Model alias, or 'all'.")
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--restricted', action='store_true', help="Use restricted dataset.")
    parser.add_argument('--perturbation-source', type=str, default='mistral', choices=['self', 'mistral'],
                        help="Perturbation source for paraphrasing. Default: mistral.")
    parser.add_argument('--filler-type', type=str, default='lorem', choices=['dots', 'lorem'],
                        help="Filler type for random partial filler. Default: lorem.")
    args = parser.parse_args()

    if args.model == 'all':
        models = sorted([
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d, 'baseline'))
        ])
    else:
        models = [args.model]

    for model in models:
        analyze_model(model, args.results_dir, args.restricted, 
                       args.perturbation_source, args.filler_type)


if __name__ == "__main__":
    main()
