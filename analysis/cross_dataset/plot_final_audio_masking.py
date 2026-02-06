# analysis/cross_dataset/plot_final_audio_masking.py

"""
This script generates cross-dataset aggregated plots for the Audio Masking experiment.

Each dataset appears as a separate line on the plot, allowing comparison of how
different content types (MMAR, Sakura variants) respond to audio degradation.

Supports:
- --mask-type: silence, noise, or 'all' (generates separate plots)
- --mask-mode: random, start, end, or 'all' (generates separate plots)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

# --- Style Guide (consistent with other cross-dataset scripts) ---
DATASET_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"},
}

MODE_LINESTYLES = {
    "random": "-",
    "start": "--",
    "end": ":",
}


def load_audio_masking_results(model_name: str, results_dir: str, dataset_name: str) -> pd.DataFrame:
    """Load all audio masking results for a dataset (all mask_type/mask_mode combinations)."""
    filename = f"audio_masking_{model_name}_{dataset_name}.jsonl"
    filepath = os.path.join(results_dir, model_name, 'audio_masking', filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results not found: {filepath}")
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(data)


def create_cross_dataset_plot(
    model_name: str,
    mask_type: str,
    mask_mode: str,
    results_dir: str,
    plots_dir: str,
    y_zoom: list = None,
    save_pdf: bool = False,
    show_ci: bool = False,
    print_line_data: bool = False,
):
    """Generate a cross-dataset plot for a specific mask_type and mask_mode."""
    
    print(f"\n--- Cross-Dataset Plot: {model_name.upper()} / {mask_type} / {mask_mode} ---")
    
    fontsize = 32
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    all_dfs = []
    
    for dataset_name, style in DATASET_STYLES.items():
        try:
            df = load_audio_masking_results(model_name, results_dir, dataset_name)
            # Filter to specific mask_type and mask_mode
            df = df[(df['mask_type'] == mask_type) & (df['mask_mode'] == mask_mode)]
            
            if df.empty:
                print(f"  - No data for {dataset_name}. Skipping.")
                continue
            
            df['dataset'] = dataset_name
            df['consistency_pct'] = df['is_consistent_with_baseline'].astype(int) * 100
            all_dfs.append(df)
            
            # Plot
            sns.lineplot(
                data=df,
                x='mask_percent',
                y='consistency_pct',
                label=style['label'],
                color=style['color'],
                marker=style['marker'],
                linestyle=MODE_LINESTYLES.get(mask_mode, '-'),
                linewidth=2,
                markersize=20,
                errorbar=('ci', 95) if show_ci else None,
                ax=ax,
                legend=False,
            )
            
            if print_line_data:
                curve = df.groupby('mask_percent')['is_consistent_with_baseline'].mean() * 100
                print(f"  {dataset_name}: X={curve.index.tolist()}, Y={[round(y,2) for y in curve.values]}")
                
        except FileNotFoundError:
            print(f"  - No data for {dataset_name}. Skipping.")
    
    if not all_dfs:
        print("  - No data found for any dataset. Aborting plot.")
        plt.close()
        return
    
    # Format plot
    title = f"Audio Masking ({mask_type.title()}, {mask_mode.title()}), {model_name.upper()}"
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('% of Audio Masked', fontsize=fontsize)
    ax.set_ylabel('Consistency (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize - 4))
    
    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)
    ax.grid(True)
    fig.tight_layout()
    
    # Save
    output_dir = os.path.join(plots_dir, model_name, 'audio_masking')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"cross_dataset_audio_masking_{model_name}_{mask_type}_{mask_mode}"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved: {png_path}")
    
    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF saved: {pdf_path}")
    
    plt.close()


def create_analysis(
    model_name: str,
    mask_type: str,
    mask_mode: str,
    results_dir: str,
    plots_dir: str,
    y_zoom: list = None,
    save_pdf: bool = False,
    show_ci: bool = False,
    print_line_data: bool = False,
):
    """Main orchestrator supporting 'all' for both mask_type and mask_mode."""
    
    mask_types = ['silence', 'noise'] if mask_type == 'all' else [mask_type]
    mask_modes = ['random', 'start', 'end'] if mask_mode == 'all' else [mask_mode]
    
    for mt in mask_types:
        for mm in mask_modes:
            create_cross_dataset_plot(
                model_name, mt, mm, results_dir, plots_dir,
                y_zoom, save_pdf, show_ci, print_line_data
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cross-dataset Audio Masking plots.")
    parser.add_argument('--model', type=str, required=True, help="Model name (qwen, salmonn, flamingo)")
    parser.add_argument('--mask-type', type=str, required=True, choices=['silence', 'noise', 'all'])
    parser.add_argument('--mask-mode', type=str, required=True, choices=['random', 'start', 'end', 'all'])
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='plots/cross_dataset_plots')
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None, help="Custom Y-axis range")
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--show-ci', action='store_true', help="Show 95% confidence interval")
    parser.add_argument('--print-line-data', action='store_true', help="Print line coordinates to console")
    
    args = parser.parse_args()
    
    create_analysis(
        args.model, args.mask_type, args.mask_mode,
        args.results_dir, args.plots_dir,
        args.y_zoom, args.save_pdf, args.show_ci, args.print_line_data
    )
