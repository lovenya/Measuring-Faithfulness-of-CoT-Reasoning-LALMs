#!/usr/bin/env python3
"""
Correct vs Incorrect Baseline Analysis

Generates separate consistency plots and summary tables for:
1. Samples where baseline was CORRECT
2. Samples where baseline was INCORRECT

For Qwen and SALMONN models, all 4 experiments, all 5 datasets, restricted mode.
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results

# Configuration
MODELS = ['qwen', 'salmonn', 'flamingo', 'salmonn_7b']
DATASETS = ['mmar', 'sakura-animal', 'sakura-emotion', 'sakura-gender', 'sakura-language']
EXPERIMENTS = {
    'adding_mistakes': {
        'x_col_calc': lambda df: ((df['mistake_position'] - 1) / df['total_sentences_in_chain']) * 100,
        'bin_size': 10,
        'xlabel': 'Percentage % of Chain Without Mistake',
        'add_100_bin': True,  # Need to add synthetic 100% bin
    },
    'early_answering': {
        'x_col_calc': lambda df: (df['num_sentences_provided'] / df['total_sentences_in_chain']) * 100,
        'bin_size': 5,
        'xlabel': 'Percentage % of Sentences Kept',
        'add_100_bin': False,
    },
    'paraphrasing': {
        'x_col_calc': lambda df: (df['num_sentences_paraphrased'] / df['total_sentences_in_chain']) * 100,
        'bin_size': 5,
        'xlabel': 'Percentage % of Sentences Paraphrased',
        'add_0_bin': True,  # Need to add synthetic 0% bin
    },
    'random_partial_filler_text': {
        'x_col_calc': lambda df: df['percent_replaced'],
        'bin_size': 5,
        'xlabel': 'Percentage % of Words Replaced',
        'force_0_consistent': True,  # 0% replaced = 100% consistent
    },
}
TARGET_X_VALUES = [0, 25, 50, 75, 100]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rebuttal_analysis', 'correct_vs_incorrect')

# Plot styles
FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"}
}


def load_experiment_data(model_name, experiment_name, dataset_name):
    """Load and prepare experiment data with baseline correctness info."""
    try:
        # Load experiment results
        exp_config = EXPERIMENTS[experiment_name]
        df = load_results(model_name, RESULTS_DIR, experiment_name, dataset_name, is_restricted=True)
        
        if df.empty:
            return None
        
        # Filter for valid chains
        if 'total_sentences_in_chain' in df.columns:
            df = df[df['total_sentences_in_chain'] > 0].copy()
        
        if df.empty:
            return None
        
        # For random_partial_filler_text, need to add consistency check
        if experiment_name == 'random_partial_filler_text':
            baseline_df = load_results(model_name, RESULTS_DIR, 'baseline', dataset_name, is_restricted=True)
            early_df = load_results(model_name, RESULTS_DIR, 'early_answering', dataset_name, is_restricted=True)
            
            baseline_predictions = baseline_df[['id', 'chain_id', 'predicted_choice', 'is_correct']].rename(
                columns={'predicted_choice': 'baseline_predicted_choice', 'is_correct': 'baseline_is_correct'})
            df = pd.merge(df, baseline_predictions, on=['id', 'chain_id'], how='inner')
            df['is_consistent_with_baseline'] = (df['predicted_choice'] == df['baseline_predicted_choice'])
            
            sentence_counts = early_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
            df = pd.merge(df, sentence_counts, on=['id', 'chain_id'], how='inner')
            df = df[df['total_sentences_in_chain'] > 0].copy()
        else:
            # Load baseline to get correctness info
            baseline_df = load_results(model_name, RESULTS_DIR, 'baseline', dataset_name, is_restricted=True)
            baseline_correct = baseline_df[['id', 'chain_id', 'is_correct']].rename(columns={'is_correct': 'baseline_is_correct'})
            df = pd.merge(df, baseline_correct, on=['id', 'chain_id'], how='inner')
        
        if df.empty:
            return None
        
        # Calculate x-axis value
        df['x_value'] = exp_config['x_col_calc'](df)
        
        # Bin the x values
        bin_size = exp_config['bin_size']
        df['x_binned'] = (df['x_value'] / bin_size).round() * bin_size
        
        df['dataset'] = dataset_name
        
        return df
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {model_name}/{experiment_name}/{dataset_name}: {e}")
        return None


def get_consistency_curve(df, experiment_name):
    """Get aggregated consistency curve."""
    exp_config = EXPERIMENTS[experiment_name]
    curve = df.groupby('x_binned')['is_consistent_with_baseline'].mean() * 100
    
    # Handle special cases
    if exp_config.get('add_100_bin'):
        curve[100] = 100.0
    if exp_config.get('add_0_bin'):
        curve[0] = 100.0
    if exp_config.get('force_0_consistent'):
        curve[0] = 100.0
    
    return curve


def interpolate_value(curve, target_x):
    """Interpolate consistency value at target x."""
    if len(curve) == 0:
        return np.nan
    
    x_vals = np.array(sorted(curve.index))
    y_vals = np.array([curve[x] for x in x_vals])
    
    if target_x in curve.index:
        return curve[target_x]
    
    if len(x_vals) < 2:
        return y_vals[0] if len(y_vals) > 0 else np.nan
    
    if target_x < x_vals.min():
        return y_vals[0]
    if target_x > x_vals.max():
        return y_vals[-1]
    
    idx = np.searchsorted(x_vals, target_x)
    x1, x2 = x_vals[idx-1], x_vals[idx]
    y1, y2 = y_vals[idx-1], y_vals[idx]
    
    return y1 + (y2 - y1) * (target_x - x1) / (x2 - x1)


def calculate_slope(curve):
    """Calculate linear regression slope."""
    if len(curve) < 2:
        return np.nan
    
    x_vals = np.array(sorted(curve.index))
    y_vals = np.array([curve[x] for x in x_vals])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    return slope


def generate_plot(super_df, model_name, experiment_name, split_name, output_dir, show_ci=False, y_zoom=None):
    """Generate and save a consistency plot."""
    exp_config = EXPERIMENTS[experiment_name]
    
    fontsize = 24
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    super_df['consistency_pct'] = super_df['is_consistent_with_baseline'].astype(int) * 100
    
    for dataset_name, style in FINAL_PLOT_STYLES.items():
        if dataset_name not in super_df['dataset'].unique():
            continue
            
        dataset_df = super_df[super_df['dataset'] == dataset_name]
        
        sns.lineplot(data=dataset_df, 
                     x='x_binned', 
                     y='consistency_pct',
                     label=style['label'], 
                     color=style['color'], 
                     marker=style['marker'], 
                     linestyle='-',
                     linewidth=2,
                     markersize=12,
                     ax=ax,
                     errorbar=('ci', 95) if show_ci else None)
    
    title = f'{experiment_name.replace("_", " ").title()}, {model_name.upper()}\n(Baseline {split_name.title()})'
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(exp_config['xlabel'], fontsize=fontsize-4)
    ax.set_ylabel('Consistency (%)', fontsize=fontsize-4)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-8))
    
    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)
    ax.legend(fontsize=fontsize-8, loc='best')
    ax.grid(True)
    fig.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(output_dir, model_name, experiment_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'{experiment_name}_{model_name}_{split_name}_restricted.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved plot: {plot_path}")


def analyze_all(filter_model=None, filter_experiment=None):
    """Run analysis for all (or filtered) combinations."""
    results = {}
    
    models_to_process = [filter_model] if filter_model else MODELS
    experiments_to_process = [filter_experiment] if filter_experiment else EXPERIMENTS.keys()
    
    for model in models_to_process:
        results[model] = {}
        for experiment in experiments_to_process:
            results[model][experiment] = {}
            for dataset in DATASETS:
                print(f"Processing: {model} / {experiment} / {dataset}")
                
                df = load_experiment_data(model, experiment, dataset)
                
                if df is None or df.empty:
                    results[model][experiment][dataset] = {'correct': None, 'incorrect': None}
                    continue
                
                result = {}
                for split_name, is_correct in [('correct', True), ('incorrect', False)]:
                    split_df = df[df['baseline_is_correct'] == is_correct].copy()
                    
                    if split_df.empty:
                        result[split_name] = None
                        continue
                    
                    curve = get_consistency_curve(split_df, experiment)
                    
                    values = {}
                    for x in TARGET_X_VALUES:
                        values[x] = interpolate_value(curve, x)
                    
                    slope = calculate_slope(curve)
                    
                    result[split_name] = {
                        'values': values,
                        'slope': slope,
                        'n_samples': len(split_df['id'].unique()),
                    }
                
                results[model][experiment][dataset] = result
    
    return results


def generate_all_plots(results, show_ci=False, y_zoom=None, filter_model=None, filter_experiment=None):
    """Generate plots for all (or filtered) model/experiment/split combinations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    models_to_process = [filter_model] if filter_model else MODELS
    experiments_to_process = [filter_experiment] if filter_experiment else EXPERIMENTS.keys()
    
    for model in models_to_process:
        for experiment in experiments_to_process:
            for split_name in ['correct', 'incorrect']:
                print(f"Generating plot: {model}/{experiment}/{split_name}")
                
                all_dfs = []
                for dataset in DATASETS:
                    df = load_experiment_data(model, experiment, dataset)
                    if df is not None and not df.empty:
                        is_correct = (split_name == 'correct')
                        split_df = df[df['baseline_is_correct'] == is_correct].copy()
                        if not split_df.empty:
                            all_dfs.append(split_df)
                
                if all_dfs:
                    super_df = pd.concat(all_dfs, ignore_index=True)
                    
                    # Add synthetic anchor points for specific experiments
                    if experiment == 'adding_mistakes':
                        # 100% of chain without mistake = 100% consistent (baseline)
                        anchor_df = super_df.drop_duplicates(subset=['id', 'chain_id', 'dataset']).copy()
                        anchor_df['x_binned'] = 100
                        anchor_df['is_consistent_with_baseline'] = True
                        super_df = pd.concat([super_df, anchor_df], ignore_index=True)
                    elif experiment == 'paraphrasing':
                        # 0% paraphrased = 100% consistent (baseline)
                        anchor_df = super_df.drop_duplicates(subset=['id', 'chain_id', 'dataset']).copy()
                        anchor_df['x_binned'] = 0
                        anchor_df['is_consistent_with_baseline'] = True
                        super_df = pd.concat([super_df, anchor_df], ignore_index=True)
                    
                    generate_plot(super_df, model, experiment, split_name, OUTPUT_DIR, show_ci=show_ci, y_zoom=y_zoom)


def format_summary_table(results):
    """Generate combined summary table with correct vs incorrect comparison."""
    lines = []
    lines.append("=" * 140)
    lines.append("CORRECT vs INCORRECT BASELINE: Consistency Analysis")
    lines.append("=" * 140)
    lines.append("")
    
    # Iterate over only the models/experiments in results (respects filtering)
    for model in results.keys():
        lines.append("")
        lines.append("#" * 140)
        lines.append(f"# MODEL: {model.upper()}")
        lines.append("#" * 140)
        
        for experiment in results[model].keys():
            lines.append("")
            lines.append("-" * 120)
            lines.append(f"Experiment: {experiment}")
            lines.append("-" * 120)
            lines.append("")
            
            # Header for comparison table
            lines.append(f"{'Dataset':<16} | {'Split':<9} | " + " | ".join([f"X={x:>3}" for x in TARGET_X_VALUES]) + " | Slope     | N")
            lines.append("-" * 100)
            
            # Collect values for averaging
            correct_slopes = []
            incorrect_slopes = []
            correct_x0 = []
            incorrect_x0 = []
            
            for dataset in DATASETS:
                data = results[model][experiment].get(dataset, {})
                
                for split_name in ['correct', 'incorrect']:
                    split_data = data.get(split_name)
                    
                    if split_data is None:
                        row = f"{dataset:<16} | {split_name:<9} | " + " | ".join(["  N/A" for _ in TARGET_X_VALUES]) + " |    N/A    | N/A"
                    else:
                        values_str = " | ".join([f"{split_data['values'][x]:>5.1f}" if not np.isnan(split_data['values'][x]) else "  N/A" for x in TARGET_X_VALUES])
                        slope_str = f"{split_data['slope']:>+8.4f}" if not np.isnan(split_data['slope']) else "    N/A "
                        n_str = f"{split_data['n_samples']:>3}"
                        row = f"{dataset:<16} | {split_name:<9} | {values_str} | {slope_str} | {n_str}"
                        
                        # Collect for averaging - use X=100 for paraphrasing/filler, X=0 for others
                        x_idx = 100 if experiment in ['paraphrasing', 'random_partial_filler_text'] else 0
                        if not np.isnan(split_data['slope']):
                            if split_name == 'correct':
                                correct_slopes.append(split_data['slope'])
                                if not np.isnan(split_data['values'][x_idx]):
                                    correct_x0.append(split_data['values'][x_idx])
                            else:
                                incorrect_slopes.append(split_data['slope'])
                                if not np.isnan(split_data['values'][x_idx]):
                                    incorrect_x0.append(split_data['values'][x_idx])
                    
                    lines.append(row)
                
                lines.append("")  # Separator between datasets
            
            # Add average row
            lines.append("-" * 100)
            x_label = "X=100" if experiment in ['paraphrasing', 'random_partial_filler_text'] else "X=0"
            lines.append(f"AVERAGES (showing {x_label} and Slope):")
            
            # Correct averages
            avg_correct_slope = np.mean(correct_slopes) if correct_slopes else np.nan
            avg_correct_x0 = np.mean(correct_x0) if correct_x0 else np.nan
            correct_slope_str = f"{avg_correct_slope:>+8.4f}" if not np.isnan(avg_correct_slope) else "    N/A "
            correct_x0_str = f"{avg_correct_x0:>5.1f}" if not np.isnan(avg_correct_x0) else "  N/A"
            
            # Incorrect averages
            avg_incorrect_slope = np.mean(incorrect_slopes) if incorrect_slopes else np.nan
            avg_incorrect_x0 = np.mean(incorrect_x0) if incorrect_x0 else np.nan
            incorrect_slope_str = f"{avg_incorrect_slope:>+8.4f}" if not np.isnan(avg_incorrect_slope) else "    N/A "
            incorrect_x0_str = f"{avg_incorrect_x0:>5.1f}" if not np.isnan(avg_incorrect_x0) else "  N/A"
            
            # Place value in correct column based on experiment type
            if experiment in ['paraphrasing', 'random_partial_filler_text']:
                # X=100 column (last X column)
                lines.append(f"{'AVERAGE':<16} | {'correct':<9} |   -   |   -   |   -   |   -   | {correct_x0_str} | {correct_slope_str} |  -")
                lines.append(f"{'AVERAGE':<16} | {'incorrect':<9} |   -   |   -   |   -   |   -   | {incorrect_x0_str} | {incorrect_slope_str} |  -")
            else:
                # X=0 column (first X column)
                lines.append(f"{'AVERAGE':<16} | {'correct':<9} | {correct_x0_str} |   -   |   -   |   -   |   -   | {correct_slope_str} |  -")
                lines.append(f"{'AVERAGE':<16} | {'incorrect':<9} | {incorrect_x0_str} |   -   |   -   |   -   |   -   | {incorrect_slope_str} |  -")
            lines.append("")
    
    return "\n".join(lines)


def main():
    print("Correct vs Incorrect Baseline Analysis")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Run analysis
    results = analyze_all()
    
    # Generate plots
    generate_all_plots(results)
    
    # Generate summary table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = format_summary_table(results)
    summary_path = os.path.join(OUTPUT_DIR, 'correct_vs_incorrect_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct vs Incorrect Baseline Analysis with CI support.")
    parser.add_argument('--model', type=str, default=None, choices=['qwen', 'salmonn', 'flamingo', 'salmonn_7b'], help="Run for specific model only.")
    parser.add_argument('--experiment', type=str, default=None, choices=['adding_mistakes', 'early_answering', 'paraphrasing', 'random_partial_filler_text'], help="Run for specific experiment only.")
    parser.add_argument('--show-ci', action='store_true', help="Show 95% confidence interval in plots.")
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None, help="Custom Y-axis range (e.g., --y-zoom 40 101).")
    args = parser.parse_args()
    
    print("Correct vs Incorrect Baseline Analysis")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    if args.show_ci:
        print("Confidence intervals: ENABLED")
    if args.y_zoom:
        print(f"Y-axis zoom: {args.y_zoom[0]} - {args.y_zoom[1]}")
    if args.model:
        print(f"Filtering to model: {args.model}")
    if args.experiment:
        print(f"Filtering to experiment: {args.experiment}")
    print()
    
    # Run analysis
    results = analyze_all(filter_model=args.model, filter_experiment=args.experiment)
    
    # Generate plots with CI and y-zoom options
    generate_all_plots(results, show_ci=args.show_ci, y_zoom=args.y_zoom, filter_model=args.model, filter_experiment=args.experiment)
    
    # Generate summary table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = format_summary_table(results)
    summary_path = os.path.join(OUTPUT_DIR, 'correct_vs_incorrect_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")
    
    print("\nDone!")
