# analysis/cross_dataset/plot_final_early_answering_100pct_words.py

"""
This script generates an alternative cross-dataset plot for the
'Early Answering' experiment.

It measures consistency against the model's own answer when 100% of the
reasoning chain has been provided, and uses Percentage of WORDS on the X-axis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import seaborn as sns
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results, discover_datasets

FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "mmau":            {"label": "MMAU",       "color": "#8c564b", "marker": "D"},
    "sakura-animal":   {"label": "S.Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "S.Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "S.Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"}
}

def count_words(text):
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())

def create_analysis(model_name: str, results_dir: str, plots_dir: str, y_zoom: list, print_line_data: bool, save_stats: bool, save_pdf: bool, show_ci: bool, restricted: bool = False):
    experiment_name = "early_answering"
    print(f"\n--- Generating Word-Based Plot for: EARLY_ANSWERING (100% Truth) ({model_name.upper()}) ---")
    
    try:
        dataset_names = discover_datasets(model_name, results_dir)
        print(f"Found datasets to process: {dataset_names}")
    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}'.")
        return

    all_dfs = []
    missing_100pct_report = []

    for dataset in dataset_names:
        # Load baseline for word counts
        try:
            baseline_df = load_results(model_name, results_dir, 'baseline', dataset, restricted=restricted)
        except FileNotFoundError:
            continue
            
        word_counts_lookup = {}
        for _, row in baseline_df.iterrows():
            cot = row.get('sanitized_cot', '')
            sentences = nltk.sent_tokenize(cot) if isinstance(cot, str) else []
            word_counts = [count_words(s) for s in sentences]
            word_counts_lookup[(row['id'], row['chain_id'])] = {
                'counts': word_counts,
                'total_words': sum(word_counts)
            }

        try:
            df = load_results(model_name, results_dir, experiment_name, dataset, restricted=restricted)
        except FileNotFoundError:
            continue
            
        if df.empty:
            continue
            
        df['dataset'] = dataset
        df = df[df['total_sentences_in_chain'] > 0].copy()

        # Isolate 100% rows
        max_per_chain = df.groupby(['id', 'chain_id'])['num_sentences_provided'].max().reset_index()
        max_per_chain.rename(columns={'num_sentences_provided': 'max_sentences'}, inplace=True)
        df_joined = df.merge(max_per_chain, on=['id', 'chain_id'])
        
        full_cot_df = df_joined[df_joined['num_sentences_provided'] == df_joined['max_sentences']]
        full_cot_truth = dict(zip(zip(full_cot_df['id'], full_cot_df['chain_id']), full_cot_df['predicted_choice']))
        
        all_samples = set(zip(df_joined['id'], df_joined['chain_id']))
        samples_with_100pct = set(full_cot_truth.keys())
        missing_samples = all_samples - samples_with_100pct
        
        if missing_samples:
            missing_100pct_report.append(f"[{dataset}] WARNING: Dropping {len(missing_samples)} question(s) missing a 100% run.")
            df_joined = df_joined[df_joined.apply(lambda row: (row['id'], row['chain_id']) in samples_with_100pct, axis=1)]

        def check_consistency(row):
            expected = full_cot_truth.get((row['id'], row['chain_id']))
            return row['predicted_choice'] == expected

        df_joined['is_consistent_with_100pct'] = df_joined.apply(check_consistency, axis=1)
        
        # Calculate percentage of WORDS
        def calculate_percent_words(row):
            info = word_counts_lookup.get((row['id'], row['chain_id']))
            if not info or info['total_words'] == 0:
                return 0.0
            n_provided = int(row['num_sentences_provided'])
            words_provided = sum(info['counts'][:n_provided])
            return (words_provided / info['total_words']) * 100.0
            
        df_joined['percent_words_provided'] = df_joined.apply(calculate_percent_words, axis=1)
        df_joined['percent_binned'] = (df_joined['percent_words_provided'] / 5).round() * 5
        
        df_joined = df_joined.drop(columns=['max_sentences'])
        all_dfs.append(df_joined)

    if missing_100pct_report:
        print("\n" + "="*70)
        print("MISSING 100% RUN REPORT")
        print("="*70)
        for line in missing_100pct_report:
            print(line)
        print("="*70 + "\n")

    if not all_dfs:
        print("No valid data remaining for any dataset. Halting analysis.")
        return
        
    super_df = pd.concat(all_dfs, ignore_index=True)

    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"cross_dataset_{experiment_name}_100pct_words_{model_name}"
    if restricted:
        base_filename += "_restricted"
    
    super_df['consistency_pct'] = super_df['is_consistent_with_100pct'].astype(int) * 100

    if print_line_data or save_stats:
        stats_output = []
        for dataset_name in sorted(super_df['dataset'].unique()):
            group_df = super_df[super_df['dataset'] == dataset_name]
            
            stats_output.append("="*60)
            stats_output.append(f"Dataset: {dataset_name} (Using 100% full CoT as truth)")
            stats_output.append("="*60)
            
            consistency_curve = group_df.groupby('percent_binned')['is_consistent_with_100pct'].mean() * 100
            stats_output.append("\nAggregated Line Data (Consistency %):")
            stats_output.append(f"  X Coords (Word %): {consistency_curve.index.tolist()}")
            stats_output.append(f"  Y Coords: {[round(y, 2) for y in consistency_curve.values.tolist()]}")
            stats_output.append("\n")

        full_stats_string = "\n".join(stats_output)
        if print_line_data:
            print(full_stats_string)
        if save_stats:
            stats_path = os.path.join(output_dir, f"{base_filename}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write(full_stats_string)

    # --- Plotting ---
    fontsize = 32
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    for dataset_name, style in FINAL_PLOT_STYLES.items():
        if dataset_name not in super_df['dataset'].unique():
            continue
            
        dataset_df = super_df[super_df['dataset'] == dataset_name]
        
        sns.lineplot(data=dataset_df, 
                     x='percent_binned', 
                     y='consistency_pct',
                     label=style['label'], 
                     color=style['color'], 
                     marker=style['marker'], 
                     linestyle='-',
                     linewidth=2,
                     markersize=20,
                     errorbar=('ci', 95) if show_ci else None,
                     ax=ax,
                     legend=False)
        
    restricted_label = " [Restricted]" if restricted else ""
    ax.set_title(f'Early Answering (100% Ref, Words){restricted_label}, {model_name.upper()}', fontsize=fontsize)
    ax.set_xlabel('Percentage % of WORDS Kept', fontsize=fontsize)
    ax.set_ylabel('Consistency with 100% Full CoT (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=(fontsize-4))
    
    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)

    ax.grid(True)
    ax.legend(title='Dataset', title_fontsize=fontsize-14, fontsize=fontsize-18, loc='best')
    fig.tight_layout()

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved successfully to: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format='pdf')
        print(f"  - PDF copy saved to: {pdf_path}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Word-based 100% baseline plots for Early Answering.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='plots/cross_dataset_plots')
    parser.add_argument('--y-zoom', nargs=2, type=float, default=None)
    parser.add_argument('--print-line-data', action='store_true')
    parser.add_argument('--save-stats', action='store_true')
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--show-ci', action='store_true')
    parser.add_argument('--restricted', action='store_true')
    args = parser.parse_args()
    
    create_analysis(args.model, args.results_dir, args.plots_dir, args.y_zoom, args.print_line_data, args.save_stats, args.save_pdf, args.show_ci, args.restricted)
