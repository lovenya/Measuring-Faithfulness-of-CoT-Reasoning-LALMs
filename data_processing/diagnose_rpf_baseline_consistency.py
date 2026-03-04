# data_processing/diagnose_rpf_baseline_consistency.py

"""
Diagnostic script to analyze prediction consistency between the
Random Partial Filler Text (0% replacement) condition and the Baseline.

Since 0% replacement means no modification to the CoT, the model should
ideally make the exact same prediction as the baseline. This script
breaks down the consistency into two cases:
1. When RPF@0% is CORRECT: How often is the baseline also correct?
2. When RPF@0% is INCORRECT: How often does the baseline make the EXACT SAME incorrect choice?

Usage:
    python data_processing/diagnose_rpf_baseline_consistency.py --model flamingo_hf --restricted
"""

import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
from utils import load_results, discover_datasets


def print_consistency_table(experiment_name: str, config_label: str, results_rows: list):
    """Helper to print the consistency table format."""
    print(f"\n{'#'*95}")
    print(f"  Experiment: {experiment_name} vs Baseline ({config_label})")
    print(f"{'#'*95}")
    
    # Table Header
    print(f"\n  {'Dataset':<20} | {'Paired':>6} | {'Cond Correct':>12} | {'-> BL Match':>11} | {'Cond Incorr.':>12} | {'-> BL Same Mistake':>18}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*12}-+-{'-'*11}-+-{'-'*12}-+-{'-'*18}")

    if not results_rows:
        print("  No suitable data available to compare.")
        return

    for r in results_rows:
        print(f"  {r['dataset']:<20} | {r['total_paired']:>6d} | {r['num_cond_correct']:>12d} | {r['bl_match_when_cond_correct']:>5d}({r['bl_match_correct_pct']:>4.1f}%) | {r['num_cond_incorrect']:>12d} | {r['bl_match_when_cond_incorrect']:>10d}({r['bl_match_incorrect_pct']:>5.1f}%)")

    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*12}-+-{'-'*11}-+-{'-'*12}-+-{'-'*18}")
    
    # Totals
    tot_paired = sum(r['total_paired'] for r in results_rows)
    tot_cond_cor = sum(r['num_cond_correct'] for r in results_rows)
    tot_cor_match = sum(r['bl_match_when_cond_correct'] for r in results_rows)
    tot_cond_incor = sum(r['num_cond_incorrect'] for r in results_rows)
    tot_incor_match = sum(r['bl_match_when_cond_incorrect'] for r in results_rows)
    
    cor_match_pct = tot_cor_match / tot_cond_cor * 100 if tot_cond_cor > 0 else 0.0
    incor_match_pct = tot_incor_match / tot_cond_incor * 100 if tot_cond_incor > 0 else 0.0
    
    print(f"  {'TOTAL':<20} | {tot_paired:>6d} | {tot_cond_cor:>12d} | {tot_cor_match:>5d}({cor_match_pct:>4.1f}%) | {tot_cond_incor:>12d} | {tot_incor_match:>10d}({incor_match_pct:>5.1f}%)")
    print()


def calculate_metrics(merged_df, dataset_name):
    total_paired = len(merged_df)
    
    # --- Analysis 1: When Condition is CORRECT ---
    cond_correct_mask = merged_df['is_correct_cond'] == True
    cond_correct_df = merged_df[cond_correct_mask]
    num_cond_correct = len(cond_correct_df)
    
    if num_cond_correct > 0:
        bl_match_when_cond_correct = (cond_correct_df['predicted_choice_cond'] == cond_correct_df['predicted_choice_bl']).sum()
        bl_match_correct_pct = bl_match_when_cond_correct / num_cond_correct * 100
    else:
        bl_match_when_cond_correct = 0
        bl_match_correct_pct = 0.0

    # --- Analysis 2: When Condition is INCORRECT ---
    cond_incorrect_mask = merged_df['is_correct_cond'] == False
    cond_incorrect_df = merged_df[cond_incorrect_mask]
    num_cond_incorrect = len(cond_incorrect_df)
    
    if num_cond_incorrect > 0:
        bl_match_when_cond_incorrect = (cond_incorrect_df['predicted_choice_cond'] == cond_incorrect_df['predicted_choice_bl']).sum()
        bl_match_incorrect_pct = bl_match_when_cond_incorrect / num_cond_incorrect * 100
    else:
        bl_match_when_cond_incorrect = 0
        bl_match_incorrect_pct = 0.0
        
    return {
        'dataset': dataset_name,
        'total_paired': total_paired,
        'num_cond_correct': num_cond_correct,
        'bl_match_when_cond_correct': bl_match_when_cond_correct,
        'bl_match_correct_pct': bl_match_correct_pct,
        'num_cond_incorrect': num_cond_incorrect,
        'bl_match_when_cond_incorrect': bl_match_when_cond_incorrect,
        'bl_match_incorrect_pct': bl_match_incorrect_pct
    }


def analyze_rpf_baseline_consistency(model: str, results_dir: str, restricted: bool, filler_type: str, perturbation_source: str):
    try:
        datasets = discover_datasets(model, results_dir)
    except FileNotFoundError:
        print(f"Model '{model}' not found in {results_dir}")
        return
        
    restricted_label = " [RESTRICTED]" if restricted else ""
    print(f"\n===============================================================================================")
    print(f"  MODEL & CONSISTENCY REPORT: {model.upper()}{restricted_label}")
    print(f"===============================================================================================")
    
    rpf_rows, par_rows, ea_rows = [], [], []

    for dataset in datasets:
        try:
            baseline_df = load_results(model, results_dir, 'baseline', dataset, restricted=restricted)
        except Exception:
            continue
            
        if baseline_df.empty:
            continue

        # ---------------------------------------------------------------------
        # 1. Random Partial Filler (0%)
        # ---------------------------------------------------------------------
        try:
            rpf_tmp = load_results(model, results_dir, 'random_partial_filler_text', dataset, filler_type=filler_type, restricted=restricted)
            rpf_0 = rpf_tmp[rpf_tmp['percent_replaced'] == 0].copy()
            merged_rpf = pd.merge(rpf_0, baseline_df, on=['id', 'chain_id'], suffixes=('_cond', '_bl'))
            if not merged_rpf.empty:
                rpf_rows.append(calculate_metrics(merged_rpf, dataset))
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 2. Paraphrasing (0%)
        # ---------------------------------------------------------------------
        try:
            par_tmp = load_results(model, results_dir, 'paraphrasing', dataset, perturbation_source=perturbation_source, restricted=restricted)
            par_0 = par_tmp[par_tmp['num_sentences_paraphrased'] == 0].copy()
            merged_par = pd.merge(par_0, baseline_df, on=['id', 'chain_id'], suffixes=('_cond', '_bl'))
            if not merged_par.empty:
                par_rows.append(calculate_metrics(merged_par, dataset))
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 3. Early Answering (100%)
        # ---------------------------------------------------------------------
        try:
            ea_tmp = load_results(model, results_dir, 'early_answering', dataset, restricted=restricted)
            ea_tmp = ea_tmp[ea_tmp['total_sentences_in_chain'] > 0]
            max_sents = ea_tmp.groupby(['id', 'chain_id'])['num_sentences_provided'].max().reset_index()
            max_sents.rename(columns={'num_sentences_provided': 'max_sentences'}, inplace=True)
            ea_joined = ea_tmp.merge(max_sents, on=['id', 'chain_id'])
            ea_100 = ea_joined[ea_joined['num_sentences_provided'] == ea_joined['max_sentences']].copy()
            
            merged_ea = pd.merge(ea_100, baseline_df, on=['id', 'chain_id'], suffixes=('_cond', '_bl'))
            if not merged_ea.empty:
                ea_rows.append(calculate_metrics(merged_ea, dataset))
        except Exception:
            pass

    print_consistency_table("Random Partial Filler (0%)", f"Filler: {filler_type.upper()}", rpf_rows)
    print_consistency_table("Paraphrasing (0%)", f"Perturbation: {perturbation_source.upper()}", par_rows)
    print_consistency_table("Early Answering (100%)", "Full CoT", ea_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction consistency across all no-modification control points vs Baseline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, help="Model alias, or 'all'.")
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--restricted', action='store_true', help="Use restricted dataset.")
    parser.add_argument('--filler-type', type=str, default='lorem', choices=['dots', 'lorem'])
    parser.add_argument('--perturbation-source', type=str, default='mistral', choices=['self', 'mistral'])
    args = parser.parse_args()

    if args.model == 'all':
        models = sorted([
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d, 'baseline'))
        ])
    else:
        models = [args.model]

    for model in models:
        analyze_rpf_baseline_consistency(model, args.results_dir, args.restricted, args.filler_type, args.perturbation_source)


if __name__ == "__main__":
    main()
