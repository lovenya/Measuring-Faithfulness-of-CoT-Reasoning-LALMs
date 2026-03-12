#!/usr/bin/env python3
"""Unified cross-dataset plotting for CoT intervention experiments."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.cot.common import (
    build_reference_lookup,
    dataset_style,
    default_reference_mode,
    discover_datasets,
    load_results,
    ordered_present_datasets,
    prepare_experiment_frame,
    reference_label,
    x_axis_label,
)

SUPPORTED_EXPERIMENTS = [
    "paraphrasing",
    "early_answering",
    "partial_filler_text",
    "flipped_partial_filler_text",
    "random_partial_filler_text",
    "adding_mistakes",
]


def _is_filler_experiment(experiment: str) -> bool:
    return experiment in {
        "partial_filler_text",
        "flipped_partial_filler_text",
        "random_partial_filler_text",
    }


def _is_perturbation_experiment(experiment: str) -> bool:
    return experiment in {"paraphrasing", "adding_mistakes"}


def _build_output_basename(
    experiment: str,
    model: str,
    reference_mode: str,
    restricted: bool,
    filler_type: str,
    perturbation_source: str,
    include_100_anchor: bool,
) -> str:
    basename = f"cross_dataset_{experiment}_{model}_{reference_mode}"

    if restricted:
        basename += "_restricted"

    if _is_filler_experiment(experiment) and filler_type != "dots":
        basename += f"_{filler_type}"

    if _is_perturbation_experiment(experiment) and perturbation_source != "self":
        basename += f"-{perturbation_source}"

    if include_100_anchor:
        basename += "_plus100anchor"

    return basename


def _add_adding_mistakes_anchor(df: pd.DataFrame) -> pd.DataFrame:
    keys = df[["id", "chain_id", "dataset"]].drop_duplicates().copy()
    if keys.empty:
        return df

    keys["progress_pct"] = 100.0
    keys["progress_binned"] = 100.0
    keys["is_consistent_with_reference"] = True
    return pd.concat([df, keys], ignore_index=True, sort=False)


def _save_stats(path: str, frames: Iterable[tuple[str, pd.DataFrame]]) -> None:
    lines: list[str] = []
    for dataset_name, frame in frames:
        curve = frame.groupby("progress_binned")["is_consistent_with_reference"].mean().mul(100).sort_index()
        lines.append("=" * 60)
        lines.append(f"Dataset: {dataset_name}")
        lines.append("=" * 60)
        lines.append(f"X: {curve.index.tolist()}")
        lines.append(f"Y: {[round(v, 2) for v in curve.values.tolist()]}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def run_analysis(args: argparse.Namespace) -> None:
    experiment = args.experiment
    reference_mode = default_reference_mode(experiment) if args.reference_mode == "auto" else args.reference_mode

    if args.include_100_anchor and experiment != "adding_mistakes":
        raise ValueError("--include-100-anchor is only supported for adding_mistakes")

    datasets = discover_datasets(args.model, args.results_dir, restricted=args.restricted)
    if not datasets:
        print("No datasets discovered from baseline files. Nothing to plot.")
        return

    print(f"Experiment: {experiment}")
    print(f"Reference mode: {reference_mode} ({reference_label(reference_mode)})")
    print(f"Datasets: {datasets}")

    all_frames: list[pd.DataFrame] = []

    for dataset in datasets:
        try:
            experiment_df = load_results(
                model_name=args.model,
                results_dir=args.results_dir,
                experiment_name=experiment,
                dataset_name=dataset,
                filler_type=args.filler_type if _is_filler_experiment(experiment) else "dots",
                perturbation_source=args.perturbation_source if _is_perturbation_experiment(experiment) else "self",
                restricted=args.restricted,
            )
        except FileNotFoundError:
            print(f"  - {dataset}: experiment file not found, skipping")
            continue

        if experiment_df.empty:
            print(f"  - {dataset}: experiment file is empty, skipping")
            continue

        working = prepare_experiment_frame(experiment_df, experiment, args.bin_size)
        if working.empty:
            print(f"  - {dataset}: no valid rows after preprocessing, skipping")
            continue

        baseline_df = None
        if reference_mode == "baseline":
            try:
                baseline_df = load_results(
                    model_name=args.model,
                    results_dir=args.results_dir,
                    experiment_name="baseline",
                    dataset_name=dataset,
                    restricted=args.restricted,
                )
            except FileNotFoundError:
                print(f"  - {dataset}: baseline file not found, skipping")
                continue

        reference_lookup = build_reference_lookup(working, reference_mode, baseline_df=baseline_df)
        if not reference_lookup:
            print(f"  - {dataset}: no reference samples for mode '{reference_mode}', skipping")
            continue

        key_series = list(zip(working["id"], working["chain_id"]))
        working["reference_predicted_choice"] = [reference_lookup.get(key) for key in key_series]
        working = working[working["reference_predicted_choice"].notna()].copy()

        if working.empty:
            print(f"  - {dataset}: all rows missing reference joins, skipping")
            continue

        working["is_consistent_with_reference"] = (
            working["predicted_choice"] == working["reference_predicted_choice"]
        )
        working["dataset"] = dataset

        if args.include_100_anchor:
            working = _add_adding_mistakes_anchor(working)

        if args.min_bin_count > 1:
            counts = working["progress_binned"].value_counts()
            valid_bins = counts[counts >= args.min_bin_count].index
            working = working[working["progress_binned"].isin(valid_bins)].copy()

        if working.empty:
            print(f"  - {dataset}: all bins filtered by min count, skipping")
            continue

        all_frames.append(working)

        total_pairs = len(set(key_series))
        kept_pairs = len(set(zip(working["id"], working["chain_id"])))
        print(
            f"  - {dataset}: rows={len(working)} | unique_pairs={kept_pairs}/{total_pairs} kept"
        )

    if not all_frames:
        print("No valid data to plot.")
        return

    super_df = pd.concat(all_frames, ignore_index=True)
    super_df["consistency_pct"] = super_df["is_consistent_with_reference"].astype(int) * 100.0

    output_dir = os.path.join(args.plots_dir, args.model, experiment)
    os.makedirs(output_dir, exist_ok=True)

    base_filename = _build_output_basename(
        experiment=experiment,
        model=args.model,
        reference_mode=reference_mode,
        restricted=args.restricted,
        filler_type=args.filler_type,
        perturbation_source=args.perturbation_source,
        include_100_anchor=args.include_100_anchor,
    )

    if args.print_line_data:
        print("\nLine coordinates (dataset-wise):")
        for dataset_name in ordered_present_datasets(super_df["dataset"].unique()):
            dataset_df = super_df[super_df["dataset"] == dataset_name]
            curve = (
                dataset_df.groupby("progress_binned")["is_consistent_with_reference"]
                .mean()
                .mul(100)
                .sort_index()
            )
            print(f"  {dataset_name}: X={curve.index.tolist()} Y={[round(v, 2) for v in curve.values.tolist()]}")

    if args.save_stats:
        stats_path = os.path.join(output_dir, f"{base_filename}_stats.txt")
        frames_for_stats = [
            (dataset_name, super_df[super_df["dataset"] == dataset_name])
            for dataset_name in ordered_present_datasets(super_df["dataset"].unique())
        ]
        _save_stats(stats_path, frames_for_stats)
        print(f"Stats written: {stats_path}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    for dataset_name in ordered_present_datasets(super_df["dataset"].unique()):
        dataset_df = super_df[super_df["dataset"] == dataset_name]
        style = dataset_style(dataset_name)

        if args.show_ci:
            sns.lineplot(
                data=dataset_df,
                x="progress_binned",
                y="consistency_pct",
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=2,
                markersize=12,
                errorbar=("ci", 95),
                ax=ax,
                label=style["label"],
            )
        else:
            curve = (
                dataset_df.groupby("progress_binned")["is_consistent_with_reference"]
                .mean()
                .mul(100)
                .sort_index()
            )
            ax.plot(
                curve.index.tolist(),
                curve.values.tolist(),
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=2,
                markersize=10,
                label=style["label"],
            )

    title_reference = reference_label(reference_mode)
    title = f"{experiment.replace('_', ' ').title()} ({title_reference} reference)"
    if args.restricted:
        title += " [restricted]"

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_axis_label(experiment), fontsize=16)
    ax.set_ylabel("Consistency (%)", fontsize=16)

    if args.y_zoom:
        ax.set_ylim(args.y_zoom[0], args.y_zoom[1])
    else:
        ax.set_ylim(-5, 105)

    ax.set_xlim(-5, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Dataset")

    fig.tight_layout()

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    fig.savefig(png_path, dpi=300)
    print(f"Plot written: {png_path}")

    if args.save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        fig.savefig(pdf_path, format="pdf")
        print(f"PDF written: {pdf_path}")

    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cross-dataset CoT consistency plots with canonical reference modes.",
    )
    parser.add_argument("--model", required=True, help="Model alias (e.g., qwen_audio_2)")
    parser.add_argument("--experiment", required=True, choices=SUPPORTED_EXPERIMENTS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plots-dir", default="plots/cot")
    parser.add_argument("--reference-mode", default="auto", choices=["auto", "baseline", "0pct", "100pct"])
    parser.add_argument("--bin-size", type=int, default=5, help="Bin size in percentage points")
    parser.add_argument("--min-bin-count", type=int, default=1, help="Drop bins with fewer rows than this")
    parser.add_argument("--restricted", action="store_true", help="Use restricted results files")
    parser.add_argument("--filler-type", default="dots", choices=["dots", "lorem"])
    parser.add_argument("--perturbation-source", default="self")
    parser.add_argument("--include-100-anchor", action="store_true", help="Add synthetic 100%% anchor (adding_mistakes only)")
    parser.add_argument("--show-ci", action="store_true", help="Use 95%% CI shaded regions")
    parser.add_argument("--print-line-data", action="store_true")
    parser.add_argument("--save-stats", action="store_true")
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--y-zoom", nargs=2, type=float, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
