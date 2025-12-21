#!/usr/bin/env python3
"""
Extract rebuttal numbers for all models, datasets, and experiments.

For each combination:
- Consistency at 0, 25, 50, 75, 100 (interpolated if needed)
- Slope of the linear fit

Output: A nice table in txt format
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory for utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_results


# Configuration
MODELS = ['qwen', 'salmonn', 'flamingo']
DATASETS = ['mmar', 'sakura-animal', 'sakura-emotion', 'sakura-gender', 'sakura-language']
EXPERIMENTS = ['adding_mistakes', 'early_answering', 'paraphrasing', 'random_partial_filler_text']
TARGET_X_VALUES = [0, 25, 50, 75, 100]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def get_experiment_x_column(experiment_name):
    """Return the x-axis column name for each experiment."""
    mapping = {
        'adding_mistakes': 'percent_before_mistake',
        'early_answering': 'percent_reasoning_provided',
        'paraphrasing': 'percent_paraphrased',
        'random_partial_filler_text': 'percent_replaced',
    }
    return mapping.get(experiment_name, 'percent_binned')


def calculate_x_axis(df, experiment_name):
    """Calculate the x-axis values based on experiment type."""
    if experiment_name == 'adding_mistakes':
        df['x_value'] = ((df['mistake_position'] - 1) / df['total_sentences_in_chain']) * 100
    elif experiment_name == 'early_answering':
        df['x_value'] = (df['num_sentences_provided'] / df['total_sentences_in_chain']) * 100
    elif experiment_name == 'paraphrasing':
        df['x_value'] = (df['num_sentences_paraphrased'] / df['total_sentences_in_chain']) * 100
    elif experiment_name == 'random_partial_filler_text':
        df['x_value'] = df['percent_replaced']
    return df


def load_and_process_data(model_name, experiment_name, dataset_name):
    """Load and process experiment data for a specific combination."""
    try:
        df = load_results(model_name, RESULTS_DIR, experiment_name, dataset_name, is_restricted=True)
        if df.empty:
            return None
        
        # Filter for valid chains
        if 'total_sentences_in_chain' in df.columns:
            df = df[df['total_sentences_in_chain'] > 0].copy()
        
        # For random_partial_filler_text, need to add consistency check
        if experiment_name == 'random_partial_filler_text':
            baseline_df = load_results(model_name, RESULTS_DIR, 'baseline', dataset_name, is_restricted=True)
            early_df = load_results(model_name, RESULTS_DIR, 'early_answering', dataset_name, is_restricted=True)
            
            baseline_predictions = baseline_df[['id', 'chain_id', 'predicted_choice']].rename(
                columns={'predicted_choice': 'baseline_predicted_choice'})
            df = pd.merge(df, baseline_predictions, on=['id', 'chain_id'], how='inner')
            df['is_consistent_with_baseline'] = (df['predicted_choice'] == df['baseline_predicted_choice'])
            
            sentence_counts = early_df[['id', 'chain_id', 'total_sentences_in_chain']].drop_duplicates()
            df = pd.merge(df, sentence_counts, on=['id', 'chain_id'], how='inner')
            df = df[df['total_sentences_in_chain'] > 0].copy()
        
        if df.empty:
            return None
            
        # Calculate x-axis
        df = calculate_x_axis(df, experiment_name)
        
        # Bin the x values
        df['x_binned'] = (df['x_value'] / 5).round() * 5
        
        return df
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {model_name}/{experiment_name}/{dataset_name}: {e}")
        return None


def get_consistency_curve(df, experiment_name):
    """Get the aggregated consistency curve."""
    curve = df.groupby('x_binned')['is_consistent_with_baseline'].mean() * 100
    
    # Handle special cases for each experiment
    if experiment_name == 'adding_mistakes':
        # 100% bin means no mistake -> 100% consistency
        curve[100] = 100.0
    elif experiment_name == 'paraphrasing':
        # 0% paraphrased -> 100% consistency
        curve[0] = 100.0
    elif experiment_name == 'random_partial_filler_text':
        # 0% replaced -> 100% consistency
        curve[0] = 100.0
    
    return curve


def interpolate_value(curve, target_x):
    """Interpolate consistency value at target x."""
    x_vals = np.array(sorted(curve.index))
    y_vals = np.array([curve[x] for x in x_vals])
    
    if target_x in curve.index:
        return curve[target_x]
    
    # Linear interpolation
    if target_x < x_vals.min():
        return y_vals[0]
    if target_x > x_vals.max():
        return y_vals[-1]
    
    # Find surrounding points
    idx = np.searchsorted(x_vals, target_x)
    x1, x2 = x_vals[idx-1], x_vals[idx]
    y1, y2 = y_vals[idx-1], y_vals[idx]
    
    # Interpolate
    return y1 + (y2 - y1) * (target_x - x1) / (x2 - x1)


def calculate_slope(curve):
    """Calculate linear regression slope."""
    x_vals = np.array(sorted(curve.index))
    y_vals = np.array([curve[x] for x in x_vals])
    
    if len(x_vals) < 2:
        return np.nan, np.nan, np.nan
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    return slope, r_value**2, p_value


def analyze_all():
    """Run analysis for all combinations and generate tables."""
    results = {}
    
    for model in MODELS:
        results[model] = {}
        for experiment in EXPERIMENTS:
            results[model][experiment] = {}
            for dataset in DATASETS:
                print(f"Processing: {model} / {experiment} / {dataset}")
                
                df = load_and_process_data(model, experiment, dataset)
                
                if df is None or df.empty:
                    results[model][experiment][dataset] = None
                    continue
                
                curve = get_consistency_curve(df, experiment)
                
                # Get interpolated values
                values = {}
                for x in TARGET_X_VALUES:
                    values[x] = interpolate_value(curve, x)
                
                # Calculate slope
                slope, r2, p_value = calculate_slope(curve)
                
                results[model][experiment][dataset] = {
                    'values': values,
                    'slope': slope,
                    'r2': r2,
                    'p_value': p_value,
                    'raw_curve': {int(k): round(v, 2) for k, v in curve.items()}
                }
    
    return results


def format_table(results, output_path):
    """Format results as a nice text table."""
    lines = []
    lines.append("=" * 120)
    lines.append("REBUTTAL ANALYSIS: Consistency Numbers and Slopes")
    lines.append("=" * 120)
    lines.append("")
    
    for model in MODELS:
        lines.append("")
        lines.append("#" * 120)
        lines.append(f"# MODEL: {model.upper()}")
        lines.append("#" * 120)
        
        for experiment in EXPERIMENTS:
            lines.append("")
            lines.append("-" * 100)
            lines.append(f"Experiment: {experiment}")
            lines.append("-" * 100)
            lines.append("")
            
            # Header
            header = f"{'Dataset':<20} | " + " | ".join([f"X={x:>3}" for x in TARGET_X_VALUES]) + " | Slope    "
            lines.append(header)
            lines.append("-" * len(header))
            
            for dataset in DATASETS:
                data = results[model][experiment].get(dataset)
                
                if data is None:
                    row = f"{dataset:<20} | " + " | ".join(["  N/A" for _ in TARGET_X_VALUES]) + " |    N/A  "
                else:
                    values_str = " | ".join([f"{data['values'][x]:>5.1f}" for x in TARGET_X_VALUES])
                    slope_str = f"{data['slope']:>+8.4f}" if not np.isnan(data['slope']) else "    N/A "
                    row = f"{dataset:<20} | {values_str} | {slope_str}"
                
                lines.append(row)
            
            lines.append("")
    
    # Write to file
    output_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"\nResults saved to: {output_path}")
    return output_text


def format_detailed_tables(results, output_dir):
    """Generate separate detailed tables for each model."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model in MODELS:
        lines = []
        lines.append("=" * 100)
        lines.append(f"MODEL: {model.upper()} - Detailed Consistency Analysis")
        lines.append("=" * 100)
        
        for experiment in EXPERIMENTS:
            lines.append("")
            lines.append("")
            lines.append("-" * 80)
            lines.append(f"EXPERIMENT: {experiment}")
            lines.append("-" * 80)
            lines.append("")
            
            # Consistency table
            lines.append("Consistency (%) at X values:")
            lines.append("")
            header = f"{'Dataset':<20}" + "".join([f" | X={x:>3}" for x in TARGET_X_VALUES])
            lines.append(header)
            lines.append("-" * len(header))
            
            for dataset in DATASETS:
                data = results[model][experiment].get(dataset)
                if data is None:
                    row = f"{dataset:<20}" + "".join([" |   N/A" for _ in TARGET_X_VALUES])
                else:
                    row = f"{dataset:<20}" + "".join([f" | {data['values'][x]:>5.1f}" for x in TARGET_X_VALUES])
                lines.append(row)
            
            lines.append("")
            lines.append("")
            lines.append("Slope Analysis:")
            lines.append("")
            lines.append(f"{'Dataset':<20} | {'Slope':>10}")
            lines.append("-" * 35)
            
            for dataset in DATASETS:
                data = results[model][experiment].get(dataset)
                if data is None:
                    lines.append(f"{dataset:<20} |        N/A")
                else:
                    slope = data['slope']
                    slope_str = f"{slope:>+10.4f}" if not np.isnan(slope) else "       N/A"
                    lines.append(f"{dataset:<20} | {slope_str}")
        
        # Save model-specific file
        output_path = os.path.join(output_dir, f"rebuttal_numbers_{model}.txt")
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Extracting rebuttal numbers...")
    print(f"Results directory: {RESULTS_DIR}")
    print()
    
    results = analyze_all()
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rebuttal_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate combined table
    combined_path = os.path.join(output_dir, 'rebuttal_numbers_combined.txt')
    format_table(results, combined_path)
    
    # Generate per-model detailed tables
    format_detailed_tables(results, output_dir)
    
    print("\nDone!")
