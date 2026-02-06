#!/usr/bin/env python3
"""
Extract rebuttal numbers for the Random Partial Filler Text experiment,
comparing Dots vs Lorem Ipsum filler types.

For Qwen and SALMONN, this script provides:
- Consistency at 0, 25, 50, 75, 100% (interpolated if needed)
- Slope of the linear fit
- Side-by-side comparison of both filler types

Output: Tables comparing dots vs lorem ipsum in txt format
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
MODELS = ['qwen', 'salmonn']
DATASETS = ['mmar', 'sakura-animal', 'sakura-emotion', 'sakura-gender', 'sakura-language']
FILLER_TYPES = ['dots', 'lorem']
TARGET_X_VALUES = [0, 25, 50, 75, 100]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def load_and_process_data(model_name, dataset_name, filler_type):
    """Load and process experiment data for a specific combination."""
    try:
        df = load_results(model_name, RESULTS_DIR, 'random_partial_filler_text', dataset_name, 
                         is_restricted=True, filler_type=filler_type)
        if df.empty:
            return None
        
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
            
        # X-axis is percent_replaced directly
        df['x_value'] = df['percent_replaced']
        df['x_binned'] = (df['x_value'] / 5).round() * 5
        
        return df
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {model_name}/{dataset_name}/{filler_type}: {e}")
        return None


def get_consistency_curve(df):
    """Get the aggregated consistency curve."""
    curve = df.groupby('x_binned')['is_consistent_with_baseline'].mean() * 100
    # 0% replaced -> 100% consistency
    curve[0] = 100.0
    return curve


def interpolate_value(curve, target_x):
    """Interpolate consistency value at target x."""
    x_vals = np.array(sorted(curve.index))
    y_vals = np.array([curve[x] for x in x_vals])
    
    if target_x in curve.index:
        return curve[target_x]
    
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
        for dataset in DATASETS:
            results[model][dataset] = {}
            for filler_type in FILLER_TYPES:
                print(f"Processing: {model} / {dataset} / {filler_type}")
                
                df = load_and_process_data(model, dataset, filler_type)
                
                if df is None or df.empty:
                    results[model][dataset][filler_type] = None
                    continue
                
                curve = get_consistency_curve(df)
                
                # Get interpolated values
                values = {}
                for x in TARGET_X_VALUES:
                    values[x] = interpolate_value(curve, x)
                
                # Calculate slope
                slope, r2, p_value = calculate_slope(curve)
                
                results[model][dataset][filler_type] = {
                    'values': values,
                    'slope': slope,
                    'r2': r2,
                    'p_value': p_value,
                    'raw_curve': {int(k): round(v, 2) for k, v in curve.items()}
                }
    
    return results


def format_combined_table(results, output_path):
    """Format results as a side-by-side comparison table."""
    lines = []
    lines.append("=" * 140)
    lines.append("REBUTTAL ANALYSIS: Random Partial Filler Text - Dots vs Lorem Ipsum Comparison")
    lines.append("=" * 140)
    lines.append("")
    lines.append("Consistency (%) at different replacement percentages")
    lines.append("Slope represents change in consistency per 1% increase in replacement")
    lines.append("")
    
    for model in MODELS:
        lines.append("")
        lines.append("#" * 140)
        lines.append(f"# MODEL: {model.upper()}")
        lines.append("#" * 140)
        lines.append("")
        
        # Header for side-by-side comparison
        header_parts = [f"{'Dataset':<20}"]
        header_parts.append("|")
        header_parts.append(f"{'DOTS':^45}")
        header_parts.append("|")
        header_parts.append(f"{'LOREM IPSUM':^45}")
        header_parts.append("|")
        header_parts.append(f"{'Δ Slope':^10}")
        lines.append("".join(header_parts))
        
        # Sub-header
        sub_parts = [f"{'':<20}"]
        for _ in range(2):
            sub_parts.append("|")
            sub_parts.append(" " + " ".join([f"X={x:>2}" for x in TARGET_X_VALUES]) + f" | {'Slope':>8}")
        sub_parts.append("|")
        sub_parts.append(f"{'(L-D)':^10}")
        lines.append("".join(sub_parts))
        
        lines.append("-" * 140)
        
        for dataset in DATASETS:
            row_parts = [f"{dataset:<20}"]
            
            slopes = {}
            for filler_type in FILLER_TYPES:
                data = results[model][dataset].get(filler_type)
                row_parts.append("|")
                
                if data is None:
                    row_parts.append(" " + " ".join(["  N/A" for _ in TARGET_X_VALUES]) + " |      N/A")
                    slopes[filler_type] = np.nan
                else:
                    values_str = " ".join([f"{data['values'][x]:>5.1f}" for x in TARGET_X_VALUES])
                    slope_str = f"{data['slope']:>+8.4f}" if not np.isnan(data['slope']) else "     N/A"
                    row_parts.append(f" {values_str} | {slope_str}")
                    slopes[filler_type] = data['slope']
            
            # Delta slope (Lorem - Dots)
            row_parts.append("|")
            if not np.isnan(slopes.get('dots', np.nan)) and not np.isnan(slopes.get('lorem', np.nan)):
                delta = slopes['lorem'] - slopes['dots']
                row_parts.append(f"{delta:^+10.4f}")
            else:
                row_parts.append(f"{'N/A':^10}")
            
            lines.append("".join(row_parts))
        
        lines.append("")
        
        # Summary statistics
        lines.append("-" * 80)
        lines.append("Summary Statistics:")
        lines.append("")
        
        for filler_type in FILLER_TYPES:
            all_slopes = []
            all_100_values = []
            for dataset in DATASETS:
                data = results[model][dataset].get(filler_type)
                if data and not np.isnan(data['slope']):
                    all_slopes.append(data['slope'])
                    all_100_values.append(data['values'][100])
            
            if all_slopes:
                lines.append(f"  {filler_type.upper():>8}: Avg Slope = {np.mean(all_slopes):+.4f}, "
                           f"Avg Consistency@100% = {np.mean(all_100_values):.1f}%")
        
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
        lines.append("=" * 120)
        lines.append(f"MODEL: {model.upper()} - Random Partial Filler: Dots vs Lorem Ipsum")
        lines.append("=" * 120)
        
        for filler_type in FILLER_TYPES:
            lines.append("")
            lines.append("")
            lines.append("-" * 80)
            lines.append(f"FILLER TYPE: {filler_type.upper()}")
            lines.append("-" * 80)
            lines.append("")
            
            # Consistency table
            lines.append("Consistency (%) at X values:")
            lines.append("")
            header = f"{'Dataset':<20}" + "".join([f" | X={x:>3}" for x in TARGET_X_VALUES])
            lines.append(header)
            lines.append("-" * len(header))
            
            for dataset in DATASETS:
                data = results[model][dataset].get(filler_type)
                if data is None:
                    row = f"{dataset:<20}" + "".join([" |   N/A" for _ in TARGET_X_VALUES])
                else:
                    row = f"{dataset:<20}" + "".join([f" | {data['values'][x]:>5.1f}" for x in TARGET_X_VALUES])
                lines.append(row)
            
            lines.append("")
            lines.append("")
            lines.append("Slope Analysis:")
            lines.append("")
            lines.append(f"{'Dataset':<20} | {'Slope':>10} | {'R²':>8} | {'p-value':>10}")
            lines.append("-" * 60)
            
            for dataset in DATASETS:
                data = results[model][dataset].get(filler_type)
                if data is None:
                    lines.append(f"{dataset:<20} |        N/A |      N/A |        N/A")
                else:
                    slope = data['slope']
                    r2 = data['r2']
                    p_val = data['p_value']
                    slope_str = f"{slope:>+10.4f}" if not np.isnan(slope) else "       N/A"
                    r2_str = f"{r2:>8.4f}" if not np.isnan(r2) else "     N/A"
                    p_str = f"{p_val:>10.4f}" if not np.isnan(p_val) else "       N/A"
                    lines.append(f"{dataset:<20} | {slope_str} | {r2_str} | {p_str}")
        
        # Save model-specific file
        output_path = os.path.join(output_dir, f"lorem_vs_dots_{model}.txt")
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Extracting Lorem vs Dots comparison numbers...")
    print(f"Results directory: {RESULTS_DIR}")
    print()
    
    results = analyze_all()
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rebuttal_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate combined comparison table
    combined_path = os.path.join(output_dir, 'lorem_vs_dots_combined.txt')
    format_combined_table(results, combined_path)
    
    # Generate per-model detailed tables
    format_detailed_tables(results, output_dir)
    
    print("\nDone!")
