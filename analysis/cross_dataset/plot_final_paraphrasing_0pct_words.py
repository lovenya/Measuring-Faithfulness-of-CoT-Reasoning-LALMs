# analysis/cross_dataset/plot_final_paraphrasing_0pct_words.py

"""
This script generates an alternative cross-dataset plot for the
'Paraphrasing' experiment.

It measures consistency against the 0% paraphrase trial from within the
paraphrasing experiment itself, mapping to Percentage of WORDS on the X-axis.
"""

import os
import argparse
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pandas as pd

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_plot_style import dataset_style, generate_dataset_legend, ordered_present_datasets
from utils import discover_datasets, load_results


def count_words(text):
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())


def create_analysis(
    model_name: str,
    results_dir: str,
    plots_dir: str,
    y_zoom: list,
    print_line_data: bool,
    save_stats: bool,
    save_pdf: bool,
    show_ci: bool,
    perturbation_source: str = "self",
    restricted: bool = False,
):
    experiment_name = "paraphrasing"
    perturbation_label = f" [{perturbation_source.upper()}]" if perturbation_source != "self" else ""
    print(
        f"\n--- Generating Word-Based Plot for: PARAPHRASING (0% Truth)"
        f"{perturbation_label} ({model_name.upper()}) ---"
    )

    try:
        dataset_names = discover_datasets(model_name, results_dir, restricted=restricted)
        print(f"Found datasets to process{perturbation_label}: {dataset_names}")
    except FileNotFoundError:
        print(f"Could not find baseline directory for model '{model_name}'.")
        return

    all_dfs = []
    missing_0pct_report = []

    for dataset in dataset_names:
        try:
            baseline_df = load_results(model_name, results_dir, "baseline", dataset, restricted=restricted)
        except FileNotFoundError:
            continue

        word_counts_lookup = {}
        for _, row in baseline_df.iterrows():
            cot = row.get("sanitized_cot", "")
            sentences = nltk.sent_tokenize(cot) if isinstance(cot, str) else []
            word_counts = [count_words(s) for s in sentences]
            word_counts_lookup[(row["id"], row["chain_id"])] = {
                "counts": word_counts,
                "total_words": sum(word_counts),
            }

        try:
            df = load_results(
                model_name,
                results_dir,
                experiment_name,
                dataset,
                perturbation_source=perturbation_source,
                restricted=restricted,
            )
        except FileNotFoundError:
            continue

        if df.empty:
            continue

        df["dataset"] = dataset

        zero_pct_df = df[df["num_sentences_paraphrased"] == 0]
        zero_pct_truth = dict(
            zip(zip(zero_pct_df["id"], zero_pct_df["chain_id"]), zero_pct_df["predicted_choice"])
        )

        all_samples = set(zip(df["id"], df["chain_id"]))
        samples_with_0pct = set(zero_pct_truth.keys())
        missing_samples = all_samples - samples_with_0pct

        if missing_samples:
            missing_0pct_report.append(
                f"[{dataset}] WARNING: Dropping {len(missing_samples)} question(s) missing a 0% run."
            )
            df = df[df.apply(lambda row: (row["id"], row["chain_id"]) in samples_with_0pct, axis=1)]
            if df.empty:
                continue

        def check_consistency(row):
            expected = zero_pct_truth.get((row["id"], row["chain_id"]))
            return row["predicted_choice"] == expected

        df["is_consistent_with_0pct"] = df.apply(check_consistency, axis=1)

        def calculate_percent_words(row):
            info = word_counts_lookup.get((row["id"], row["chain_id"]))
            if not info or info["total_words"] == 0:
                return 0.0
            n_para = int(row["num_sentences_paraphrased"])
            words_para = sum(info["counts"][:n_para])
            return (words_para / info["total_words"]) * 100.0

        df["percent_words_paraphrased"] = df.apply(calculate_percent_words, axis=1)
        df["percent_binned"] = (df["percent_words_paraphrased"] / 5).round() * 5

        # Filter out sparse bins (less than 15 samples)
        bin_counts = df["percent_binned"].value_counts()
        valid_bins = bin_counts[bin_counts >= 15].index
        df = df[df["percent_binned"].isin(valid_bins)]

        all_dfs.append(df)

    if missing_0pct_report:
        print("\n" + "=" * 70)
        print("MISSING 0% RUN REPORT")
        print("=" * 70)
        for line in missing_0pct_report:
            print(line)
        print("=" * 70 + "\n")

    if not all_dfs:
        print("No valid data remaining for any dataset. Halting analysis.")
        return

    super_df = pd.concat(all_dfs, ignore_index=True)

    output_dir = os.path.join(plots_dir, model_name, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"cross_dataset_{experiment_name}_0pct_words_{model_name}"
    if restricted:
        base_filename += "_restricted"
    if perturbation_source != "self":
        base_filename += f"-{perturbation_source}"

    super_df["consistency_pct"] = super_df["is_consistent_with_0pct"].astype(int) * 100

    if print_line_data or save_stats:
        stats_output = []
        for dataset_name in sorted(super_df["dataset"].unique()):
            group_df = super_df[super_df["dataset"] == dataset_name]

            stats_output.append("=" * 60)
            stats_output.append(f"Dataset: {dataset_name} (Using 0% un-paraphrased CoT as truth)")
            stats_output.append("=" * 60)

            consistency_curve = group_df.groupby("percent_binned")["is_consistent_with_0pct"].mean() * 100
            stats_output.append("\nAggregated Line Data (Consistency %):")
            stats_output.append(f"  X Coords (Word %): {consistency_curve.index.tolist()}")
            stats_output.append(f"  Y Coords: {[round(y, 2) for y in consistency_curve.values.tolist()]}")
            stats_output.append("\n")

        full_stats_string = "\n".join(stats_output)
        if print_line_data:
            print(full_stats_string)
        if save_stats:
            stats_path = os.path.join(output_dir, f"{base_filename}_stats.txt")
            with open(stats_path, "w") as f:
                f.write(full_stats_string)

    plt.figure(figsize=(14, 10))

    for dataset_name in ordered_present_datasets(super_df["dataset"].unique()):
        dataset_df = super_df[super_df["dataset"] == dataset_name]
        style = dataset_style(dataset_name)
        
        if show_ci:
            dataset_df_copy = dataset_df.copy()
            dataset_df_copy["consistency_pct"] = dataset_df_copy["is_consistent_with_0pct"].astype(int) * 100
            sns.lineplot(
                data=dataset_df_copy,
                x="percent_binned",
                y="consistency_pct",
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                errorbar=("ci", 95),
                ax=plt.gca(),
            )
        else:
            consistency_curve = (
                dataset_df.groupby("percent_binned")["is_consistent_with_0pct"]
                .mean()
                .mul(100)
                .sort_index()
            )
            plt.plot(
                consistency_curve.index.tolist(),
                consistency_curve.values.tolist(),
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
            )

    title_suffix = " [Mistral]" if perturbation_source == 'mistral' else ""
    restricted_label = " [Restricted]" if restricted else ""
    model_display = "Audio Flamingo 3" if model_name == "flamingo_hf" else "Qwen 2.5 Omni" if model_name == "qwen_omni" else model_name.upper()
    plt.title(f'Paraphrasing, {model_display}')
    plt.xlabel('Percentage (%) of Sentences Paraphrased', fontsize=38)
    plt.ylabel('Consistency (%)')

    if y_zoom:
        plt.ylim(y_zoom[0], y_zoom[1])
    else:
        plt.ylim(-5, 105)
    plt.xlim(-5, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  - Plot saved successfully to: {png_path}")

    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"  - PDF copy saved to: {pdf_path}")
    plt.close()

    legend_path = generate_dataset_legend(output_dir, super_df["dataset"].unique())
    print(f"  - Standalone dataset legend saved to: {legend_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Word-based 0% baseline plots for Paraphrasing.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--plots_dir", type=str, default="plots/cross_dataset_plots")
    parser.add_argument("--y-zoom", nargs=2, type=float, default=None)
    parser.add_argument("--print-line-data", action="store_true")
    parser.add_argument("--save-stats", action="store_true")
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--show-ci", action="store_true")
    parser.add_argument("--perturbation-source", type=str, default="self", choices=["self", "mistral"])
    parser.add_argument("--restricted", action="store_true")
    args = parser.parse_args()

    create_analysis(
        args.model,
        args.results_dir,
        args.plots_dir,
        args.y_zoom,
        args.print_line_data,
        args.save_stats,
        args.save_pdf,
        args.show_ci,
        args.perturbation_source,
        args.restricted,
    )
