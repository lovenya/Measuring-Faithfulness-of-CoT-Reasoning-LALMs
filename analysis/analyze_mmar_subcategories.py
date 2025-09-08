# analysis/analyze_mmar_subcategories.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import librosa
import logging
from .utils import load_results

def setup_logging(condition: str, dataset_name: str):
    """
    This helper function sets up our logging system for the analysis.
    Instead of just printing a wall of text to the terminal, we'll save a clean,
    permanent log file for each run. It's great for keeping records and makes
    the terminal output much cleaner.
    """
    # We'll create a dedicated 'logs' directory to keep our project root tidy.
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # We create a clear, timestamped filename. This is a nice touch to make sure
    # we never accidentally overwrite the log from a previous run.
    log_filename = f"subcategory_analysis_{dataset_name}_{condition}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Here, we configure the logger. We tell it to write to both the file and the console,
    # so we can see what's happening live while also saving the details.
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', # A simple format, as we just want the print output.
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging analysis details to: {log_filepath}\n")

def create_subcategory_plot(df_baseline: pd.DataFrame, df_no_reasoning: pd.DataFrame, breakdown_col: str, dataset_name: str, plots_dir: str, condition: str):
    """
    This is the main workhorse function for plotting. It takes our two main datasets,
    figures out how to break them down by a specific category (like 'modality' or 'duration'),
    calculates all the necessary stats, and then generates and saves a beautiful grouped bar chart.
    """
    logging.info(f"\n--- Analyzing breakdown by: '{breakdown_col}' ---")

    # First, we get a sorted list of all the unique categories we're going to plot.
    # This ensures our x-axis is always in a predictable, alphabetical (or numerical) order.
    all_categories = sorted(df_baseline[breakdown_col].dropna().unique())
    
    # We'll build up a list of dictionaries with our results. This is a clean and
    # easy way to structure the data before we hand it off to pandas and seaborn for plotting.
    plot_data = []

    # Now, we loop through each category one by one to calculate its specific stats.
    for category in all_categories:
        # For each category, we create a "mini-dataset" containing only the relevant rows.
        bl_cat_df = df_baseline[df_baseline[breakdown_col] == category]
        nr_cat_df = df_no_reasoning[df_no_reasoning[breakdown_col] == category]

        # --- Null/Refusal Logging ---
        # This is a crucial diagnostic step. The final accuracy number doesn't tell the whole story.
        # We also want to know if one condition is simply refusing to answer more often than the other.
        bl_valid = bl_cat_df.dropna(subset=['predicted_choice'])
        bl_valid = bl_valid[bl_valid['predicted_choice'] != "REFUSAL"]
        bl_refusals = len(bl_cat_df) - len(bl_valid)
        
        nr_valid = nr_cat_df.dropna(subset=['predicted_choice'])
        nr_valid = nr_valid[nr_valid['predicted_choice'] != "REFUSAL"]
        nr_refusals = len(nr_cat_df) - len(nr_valid)

        # The 'category' for our duration bins is a special 'Interval' object from pandas.
        # We need to check for this and format it into a nice, human-readable string for our logs and plot labels.
        category_name = f"{category.left:.2f}s - {category.right:.2f}s" if isinstance(category, pd.Interval) else category
        
        logging.info(f"  Category: {category_name}")

        # This is our safety check to prevent the ZeroDivisionError if a category happens to be empty for one experiment.
        if len(bl_cat_df) > 0:
            logging.info(f"    Baseline:      {bl_refusals}/{len(bl_cat_df)} ({bl_refusals/len(bl_cat_df):.2%}) null/refusal answers.")
        else:
            logging.info(f"    Baseline:      0/0 (N/A) null/refusal answers (No chains in this category).")

        if len(nr_cat_df) > 0:
            logging.info(f"    No-Reasoning:  {nr_refusals}/{len(nr_cat_df)} ({nr_refusals/len(nr_cat_df):.2%}) null/refusal answers.")
        else:
            logging.info(f"    No-Reasoning:  0/0 (N/A) null/refusal answers (No chains in this category).")

        # --- Accuracy Calculation (Nulls Excluded) ---
        # Here, we apply our robust macro-averaging logic. We group by each unique question 'id'
        # first to get the average accuracy for that question, and then we average those averages.
        # This is a key methodological choice to prevent questions with more chains from unfairly dominating the result.
        bl_accuracy = bl_valid.groupby('id')['is_correct'].mean().mean() * 100 if not bl_valid.empty else 0
        nr_accuracy = nr_valid.groupby('id')['is_correct'].mean().mean() * 100 if not nr_valid.empty else 0
        
        # We add the results for this category to our list.
        plot_data.append({'Category': category_name, 'Accuracy': bl_accuracy, 'Experiment': 'Baseline'})
        plot_data.append({'Category': category_name, 'Accuracy': nr_accuracy, 'Experiment': 'No-Reasoning'})

    plot_df = pd.DataFrame(plot_data)

    # --- Plotting ---
    # Now that the data is prepared, we can create the visualization.
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Seaborn's barplot is perfect for this kind of grouped comparison.
    sns.barplot(data=plot_df, x='Category', y='Accuracy', hue='Experiment', ax=ax, palette='viridis')

    # A good plot needs a clear title and labels to tell its story.
    ax.set_title(f'Baseline vs. No-Reasoning Accuracy by {breakdown_col.replace("_", " ").title()}\n({dataset_name.upper()} - {condition.replace("_", " ").title()})', fontsize=16, pad=20)
    ax.set_xlabel(breakdown_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel('Macro-Averaged Accuracy (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10) # Rotate labels to prevent them from overlapping.
    ax.set_ylim(0, 100)
    ax.legend(title='Experiment')
    fig.tight_layout() # This handy function adjusts plot params for a tight layout.

    # --- Saving ---
    # We save the plot to a clean, organized directory structure that includes the condition.
    output_plot_dir = os.path.join(plots_dir, 'subcategory_analysis', dataset_name, condition)
    os.makedirs(output_plot_dir, exist_ok=True)
    plot_path = os.path.join(output_plot_dir, f"accuracy_by_{breakdown_col}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"    Plot saved to: {plot_path}")


def main():
    """
    This is the main orchestrator for our subcategory analysis. It handles user input,
    loads the data, enriches it with metadata, and then calls the plotting function
    for each requested breakdown.
    """
    parser = argparse.ArgumentParser(description="Generate subcategory analysis plots for the MMAR dataset.")
    parser.add_argument('--dataset', type=str, default='mmar', help="The dataset to analyze (must be 'mmar').")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--plots_dir', type=str, default='./plots')
    parser.add_argument('--condition', type=str, default='default', choices=['default', 'transcribed_audio'], help="The experimental condition to analyze.")
    parser.add_argument('--breakdown-by', type=str, default='all', choices=['all', 'modality', 'category', 'sub-category', 'duration'], help="The column to break down the analysis by.")
    args = parser.parse_args()

    # We set up the logger right at the start of the program.
    setup_logging(args.condition, args.dataset)

    if args.dataset != 'mmar':
        logging.info("ERROR: This analysis script is specifically designed for the MMAR dataset's subcategories.")
        return

    logging.info(f"\n--- Starting MMAR Subcategory Analysis for Condition: '{args.condition}' ---")
    
    try:
        # --- Step 1: Load the Raw Results and Metadata ---
        # First, we load the results from our experiments. These files contain the model's
        # performance data but are missing the rich metadata we need for the breakdown.
        baseline_df = load_results(args.results_dir, 'baseline', args.dataset, args.condition)
        no_reasoning_df = load_results(args.results_dir, 'no_reasoning', args.dataset, args.condition)
        
        # Next, we load our "master" standardized file. This is the one we created with
        # the enrichment script, and it contains all the metadata.
        standardized_file_path = f'data/{args.dataset}/{args.dataset}_test_standardized.jsonl'
        logging.info(f"Loading metadata from '{standardized_file_path}'...")
        metadata_df = pd.read_json(standardized_file_path, lines=True)
        
        # We only need a few key columns from the metadata file for our merge. This is also
        # where we prevent the 'audio_path' column collision we discovered earlier.
        metadata_cols = ['id', 'modality', 'category', 'sub-category']
        metadata_df = metadata_df[[col for col in metadata_cols if col in metadata_df.columns]]

    except FileNotFoundError:
        # If any of the essential files are missing, we can't proceed.
        return

    # --- Step 2: Enrich the Results DataFrames ---
    # This is the crucial step you identified. We perform a 'left merge' to add the
    # metadata columns to our results dataframes, matching on the unique question 'id'.
    logging.info("Enriching results data with metadata...")
    baseline_df = pd.merge(baseline_df, metadata_df, on='id', how='left')
    no_reasoning_df = pd.merge(no_reasoning_df, metadata_df, on='id', how='left')
    
    breakdown_columns = []
    if args.breakdown_by == 'all':
        breakdown_columns = ['modality', 'category', 'sub-category', 'duration']
    else:
        breakdown_columns = [args.breakdown_by]

    # --- Duration Calculation Block ---
    # This is a special case. If the user wants to analyze by duration, we first need to
    # calculate the duration of each audio file, as it's not in our JSONL files.
    if 'duration' in breakdown_columns:
        logging.info("\nCalculating audio durations for breakdown...")
        # We get a unique list of audio paths to avoid wastefully loading the same file multiple times.
        unique_audio_paths = baseline_df['audio_path'].unique()
        duration_map = {}
        for i, path in enumerate(unique_audio_paths):
            # A little progress indicator for this potentially long step.
            print(f"  - Loading audio {i+1}/{len(unique_audio_paths)}: {os.path.basename(str(path))}", end='\r')
            try:
                # The 'transcribed_audio' condition is text-only, so we can't get a duration.
                if args.condition == 'transcribed_audio':
                    duration_map[path] = -1 # Use a dummy value to indicate it's not applicable.
                else:
                    duration_map[path] = librosa.get_duration(path=path)
            except Exception as e:
                print(f"\nCould not get duration for {path}: {e}")
                duration_map[path] = -1
        print("\nDuration calculation complete.")

        # Now we map these calculated durations back into our main dataframes.
        baseline_df['duration'] = baseline_df['audio_path'].map(duration_map)
        no_reasoning_df['duration'] = no_reasoning_df['audio_path'].map(duration_map)
        
        # If we're in the transcribed condition, we can't proceed with duration analysis.
        if args.condition == 'transcribed_audio':
            logging.info("  - NOTE: Duration analysis is not applicable for the 'transcribed_audio' condition. Skipping.")
            breakdown_columns.remove('duration')
        else:
            # Here's the robust part: we use `qcut` to create 10 quantile-based bins (deciles).
            # This is scientifically better than fixed-width bins because it guarantees each bin
            # has an equal number of samples, making our statistics reliable.
            try:
                # 1. We create the bins based on the baseline data ONLY. This gives us
                #    a single, authoritative set of bin edges.
                baseline_df['duration_bin'], bin_edges = pd.qcut(baseline_df['duration'], 10, duplicates='drop', retbins=True)
                
                # 2. We then apply these EXACT SAME bin edges to the no_reasoning data.
                #    This guarantees that a sample with a given duration will fall into the
                #    exact same bin in both dataframes.
                no_reasoning_df['duration_bin'] = pd.cut(no_reasoning_df['duration'], bins=bin_edges, include_lowest=True)

                # This is crucial for interpretability. We need to know what "Bin 5" actually means.
                logging.info("\n--- Audio Duration Bin Ranges (Deciles) ---")
                bin_ranges = baseline_df['duration_bin'].unique().sort_values()
                for i, bin_range in enumerate(bin_ranges):
                    # This is the fix for the AttributeError, checking the type first.
                    if isinstance(bin_range, pd.Interval):
                        logging.info(f"  Bin {i+1}: {bin_range.left:.2f}s to {bin_range.right:.2f}s")
                    else:
                        logging.info(f"  Bin {i+1}: Represents a single duration point around {bin_range:.2f}s")
                logging.info("------------------------------------------")

            except ValueError as e:
                logging.info(f"Could not create 10 duration bins, likely due to insufficient unique duration values: {e}")
                breakdown_columns.remove('duration')

    # The main loop that calls our plotting function for each requested breakdown.
    for col in breakdown_columns:
        # For the duration breakdown, we use the new 'duration_bin' column we just created.
        breakdown_col = 'duration_bin' if col == 'duration' else col
        
        if breakdown_col not in baseline_df.columns:
            logging.info(f"WARNING: Breakdown column '{breakdown_col}' not found in the data. Skipping.")
            continue
        create_subcategory_plot(baseline_df, no_reasoning_df, breakdown_col, args.dataset, args.plots_dir, args.condition)


if __name__ == "__main__":
    main()