# analysis/per_dataset/plot_audio_masking.py

"""
This script generates plots for the Audio Masking experiment.

The scientific goal is to determine how much the model relies on the audio input
for its reasoning. By progressively masking portions of the audio (with silence
or noise, from different positions), we measure how consistency degrades.

This script supports:
- --mask-type: silence, noise, or 'all' (generates separate plots for each)
- --mask-mode: random, start, end, or 'all' (plots all modes as separate lines)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json

# Add parent directory for utils import (if needed in future)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Line styles for different mask modes when --mask-mode all
MODE_STYLES = {
    "scattered": {"label": "Scattered", "color": "#e41a1c", "linestyle": "-", "marker": "o"},
    "start":     {"label": "From Start", "color": "#377eb8", "linestyle": "--", "marker": "s"},
    "end":       {"label": "From End", "color": "#4daf4a", "linestyle": ":", "marker": "^"},
}


def load_audio_masking_results(model_name: str, results_dir: str, dataset_name: str, mask_type: str, mask_mode: str) -> pd.DataFrame:
    """Load audio masking results JSONL file from hierarchical directory structure."""
    # Hierarchical path: results/{model}/audio_masking/{mask_type}/{mask_mode}/
    filename = f"audio_masking_{model_name}_{dataset_name}_{mask_type}_{mask_mode}.jsonl"
    filepath = os.path.join(results_dir, model_name, 'audio_masking', mask_type, mask_mode, filename)
    
    # Fallback: try flat structure (old combined file) for backwards compatibility
    if not os.path.exists(filepath):
        flat_filepath = os.path.join(results_dir, model_name, 'audio_masking', filename)
        if os.path.exists(flat_filepath):
            filepath = flat_filepath
        else:
            # Try old combined file with filtering
            old_filename = f"audio_masking_{model_name}_{dataset_name}.jsonl"
            old_filepath = os.path.join(results_dir, model_name, 'audio_masking', old_filename)
            if os.path.exists(old_filepath):
                data = []
                with open(old_filepath, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get('mask_type') == mask_type and entry.get('mask_mode') == mask_mode:
                                data.append(entry)
                        except json.JSONDecodeError:
                            continue
                return pd.DataFrame(data)
            raise FileNotFoundError(f"Results not found: {filepath}")
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(data)


def plot_single_graph(
    df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    mask_type: str,
    mask_mode: str,
    plots_dir: str,
    save_as_pdf: bool = False,
):
    """Generate and save a single audio masking plot."""
    
    if df.empty:
        print(f"  - No data for {mask_type}/{mask_mode}. Skipping.")
        return
    
    # Calculate consistency at each mask level
    consistency_curve = df.groupby('mask_percent')['is_consistent_with_baseline'].mean() * 100
    consistency_curve.sort_index(inplace=True)
    
    num_samples = len(df['id'].unique())
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))
    
    style = MODE_STYLES.get(mask_mode, MODE_STYLES["scattered"])
    ax.plot(consistency_curve.index, consistency_curve.values, 
            marker=style["marker"], linestyle=style["linestyle"], 
            color=style["color"], label=f'Consistency ({mask_mode})')
    
    ax.set_title(f'Audio Masking ({mask_type.title()}, {mask_mode.title()}) - {model_name.upper()} on {dataset_name.upper()}\n({num_samples} samples)', 
                 fontsize=16, pad=20)
    ax.set_xlabel('% of Audio Masked', fontsize=12)
    ax.set_ylabel('Consistency with Baseline (%)', fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc='best')
    fig.tight_layout()
    
    # --- Save ---
    output_dir = os.path.join(plots_dir, model_name, 'audio_masking', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"audio_masking_{model_name}_{dataset_name}_{mask_type}_{mask_mode}"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved: {png_path}")
    
    if save_as_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF saved: {pdf_path}")
    
    plt.close()


def plot_all_modes(
    model_name: str,
    dataset_name: str,
    mask_type: str,
    results_dir: str,
    plots_dir: str,
    save_as_pdf: bool = False,
):
    """Plot all mask modes on a single chart for comparison."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 8))
    
    total_samples = 0
    
    for mode, style in MODE_STYLES.items():
        try:
            df = load_audio_masking_results(model_name, results_dir, dataset_name, mask_type, mode)
            if df.empty:
                continue
            
            # Consistency curve (solid lines)
            consistency_curve = df.groupby('mask_percent')['is_consistent_with_baseline'].mean() * 100
            consistency_curve.sort_index(inplace=True)
            
            ax.plot(consistency_curve.index, consistency_curve.values,
                    marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], label=f'{style["label"]} (Consistency)',
                    linewidth=2, markersize=8)
            
            # Accuracy curve (dashed lines, same color but lighter)
            if 'is_correct' in df.columns:
                accuracy_curve = df.groupby('mask_percent')['is_correct'].mean() * 100
                accuracy_curve.sort_index(inplace=True)
                
                ax.plot(accuracy_curve.index, accuracy_curve.values,
                        marker=style["marker"], linestyle='--',
                        color=style["color"], label=f'{style["label"]} (Accuracy)',
                        linewidth=1.5, markersize=6, alpha=0.6)
            
            total_samples = max(total_samples, len(df['id'].unique()))
        except FileNotFoundError:
            print(f"  - No data for mode '{mode}'. Skipping.")
    
    ax.set_title(f'Audio Masking ({mask_type.title()}) - All Modes Comparison\n{model_name.upper()} on {dataset_name.upper()} ({total_samples} samples)', 
                 fontsize=16, pad=20)
    ax.set_xlabel('% of Audio Masked', fontsize=12)
    ax.set_ylabel('Consistency with Baseline (%)', fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.legend(title='Mask Mode', loc='best')
    fig.tight_layout()
    
    # --- Save ---
    output_dir = os.path.join(plots_dir, model_name, 'audio_masking', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"audio_masking_{model_name}_{dataset_name}_{mask_type}_all_modes"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved: {png_path}")
    
    if save_as_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF saved: {pdf_path}")
    
    plt.close()


def create_analysis(
    model_name: str,
    dataset_name: str,
    mask_type: str,
    mask_mode: str,
    results_dir: str,
    plots_dir: str,
    save_as_pdf: bool = False,
):
    """Main analysis orchestrator."""
    
    mask_types = ['silence', 'noise'] if mask_type == 'all' else [mask_type]
    
    for mt in mask_types:
        print(f"\n--- Generating Audio Masking Plot: {model_name.upper()} / {dataset_name} / {mt} ---")
        
        if mask_mode == 'all':
            plot_all_modes(model_name, dataset_name, mt, results_dir, plots_dir, save_as_pdf)
        else:
            try:
                df = load_audio_masking_results(model_name, results_dir, dataset_name, mt, mask_mode)
                plot_single_graph(df, model_name, dataset_name, mt, mask_mode, plots_dir, save_as_pdf)
            except FileNotFoundError as e:
                print(f"  - {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Audio Masking plots.")
    parser.add_argument('--model', type=str, required=True, help="Model name (qwen, salmonn, flamingo)")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name or 'all'")
    parser.add_argument('--mask-type', type=str, required=True, choices=['silence', 'noise', 'all'])
    parser.add_argument('--mask-mode', type=str, required=True, choices=['scattered', 'start', 'end', 'all'])
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--save-pdf', action='store_true')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = ['mmar', 'sakura-animal', 'sakura-emotion', 'sakura-gender', 'sakura-language']
        for ds in datasets:
            create_analysis(args.model, ds, args.mask_type, args.mask_mode, 
                           args.results_dir, args.plots_dir, args.save_pdf)
    else:
        create_analysis(args.model, args.dataset, args.mask_type, args.mask_mode,
                       args.results_dir, args.plots_dir, args.save_pdf)
