# analysis/per_dataset/plot_audio_masking.py

"""Generate per-dataset plots for the partial audio masking experiment.

Consistency is computed at analysis time by joining experiment outputs with
baseline predictions on (id, chain_id).
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory for utils import (if needed in future)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODE_STYLES = {
    "scattered": {"label": "Scattered", "color": "#e41a1c", "linestyle": "-", "marker": "o"},
    "start": {"label": "From Start", "color": "#377eb8", "linestyle": "--", "marker": "s"},
    "end": {"label": "From End", "color": "#4daf4a", "linestyle": ":", "marker": "^"},
}


def get_hop_type(entry: dict) -> str | None:
    hop = entry.get("hop_type")
    if hop:
        return hop
    sample_id = entry.get("id", "")
    if str(sample_id).endswith("_single"):
        return "single"
    if str(sample_id).endswith("_multi"):
        return "multi"
    return None


def filter_by_hop_type(df: pd.DataFrame, hop_type: str) -> pd.DataFrame:
    if hop_type == "merged" or df.empty:
        return df
    mask = df.apply(lambda row: get_hop_type(row.to_dict()) == hop_type, axis=1)
    return df[mask].reset_index(drop=True)


def _candidate_modes(mask_mode: str) -> list[str]:
    # Legacy compatibility: old runs may have been saved with `random`.
    if mask_mode == "scattered":
        return ["scattered", "random"]
    return [mask_mode]


def _load_jsonl(filepath: str) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_baseline_lookup(model_name: str, results_dir: str, dataset_name: str) -> dict[tuple[str, str], str | None]:
    path = os.path.join(
        results_dir,
        model_name,
        "baseline",
        f"baseline_{model_name}_{dataset_name}.jsonl",
    )
    if not os.path.exists(path):
        return {}

    lookup = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lookup[(str(row.get("id")), str(row.get("chain_id")))] = row.get("predicted_choice")
    return lookup


def _attach_consistency(
    df: pd.DataFrame,
    model_name: str,
    results_dir: str,
    dataset_name: str,
) -> pd.DataFrame:
    baseline_lookup = _load_baseline_lookup(model_name, results_dir, dataset_name)
    baseline_preds = []
    consistency_values = []

    for _, row in df.iterrows():
        key = (str(row.get("id")), str(row.get("chain_id")))
        baseline_pred = baseline_lookup.get(key)
        baseline_preds.append(baseline_pred)

        if baseline_pred is None:
            consistency_values.append(float("nan"))
        else:
            consistency_values.append(1.0 if row.get("predicted_choice") == baseline_pred else 0.0)

    df = df.copy()
    df["corresponding_baseline_predicted_choice"] = baseline_preds
    df["consistency_value"] = consistency_values
    df["is_consistent_with_baseline"] = df["consistency_value"].map({1.0: True, 0.0: False})

    missing = int(df["corresponding_baseline_predicted_choice"].isna().sum())
    df.attrs["missing_baseline_count"] = missing
    return df


def load_audio_masking_results(
    model_name: str,
    results_dir: str,
    dataset_name: str,
    mask_type: str,
    mask_mode: str,
) -> pd.DataFrame:
    """Load canonical partial_audio_masking results with legacy fallbacks."""
    candidate_files = []

    for mode in _candidate_modes(mask_mode):
        candidate_files.append(
            os.path.join(
                results_dir,
                model_name,
                "partial_audio_masking",
                mask_type,
                mode,
                f"partial_audio_masking_{model_name}_{dataset_name}_{mask_type}_{mode}.jsonl",
            )
        )

    for mode in _candidate_modes(mask_mode):
        legacy_name = f"audio_masking_{model_name}_{dataset_name}_{mask_type}_{mode}.jsonl"
        candidate_files.append(
            os.path.join(results_dir, model_name, "audio_masking", mask_type, mode, legacy_name)
        )
        candidate_files.append(
            os.path.join(results_dir, model_name, "audio_masking", legacy_name)
        )

    # Old combined legacy file.
    candidate_files.append(
        os.path.join(results_dir, model_name, "audio_masking", f"audio_masking_{model_name}_{dataset_name}.jsonl")
    )

    for filepath in candidate_files:
        if not os.path.exists(filepath):
            continue

        rows = _load_jsonl(filepath)
        df = pd.DataFrame(rows)

        # Filter old combined file rows when needed.
        if filepath.endswith(f"audio_masking_{model_name}_{dataset_name}.jsonl"):
            if "mask_type" not in df.columns or "mask_mode" not in df.columns:
                continue
            df = df[(df["mask_type"] == mask_type) & (df["mask_mode"].isin(_candidate_modes(mask_mode)))]

        if df.empty:
            continue

        df = _attach_consistency(df, model_name, results_dir, dataset_name)
        df.attrs["source_path"] = filepath
        return df

    raise FileNotFoundError("No partial_audio_masking/audio_masking results found for requested filters.")


def plot_single_graph(
    df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    mask_type: str,
    mask_mode: str,
    plots_dir: str,
    save_as_pdf: bool = False,
    hop_type: str = "merged",
):
    if df.empty:
        print(f"  - No data for {mask_type}/{mask_mode}. Skipping.")
        return

    consistency_curve = df.groupby("mask_percent")["consistency_value"].mean() * 100
    consistency_curve.sort_index(inplace=True)

    num_samples = len(df["id"].unique())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 8))

    style = MODE_STYLES.get(mask_mode, MODE_STYLES["scattered"])
    ax.plot(
        consistency_curve.index,
        consistency_curve.values,
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
        label=f"Consistency ({mask_mode})",
    )

    ax.set_title(
        "Partial Audio Masking "
        f"({mask_type.title()}, {mask_mode.title()}) - {model_name.upper()} on {dataset_name.upper()}\n"
        f"({num_samples} samples)",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("% of Audio Masked", fontsize=12)
    ax.set_ylabel("Consistency with Baseline (%)", fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best")
    fig.tight_layout()

    output_dir = os.path.join(plots_dir, model_name, "partial_audio_masking", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    base_filename = f"partial_audio_masking_{model_name}_{dataset_name}_{mask_type}_{mask_mode}"

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format="pdf")
        print(f"  - PDF saved: {pdf_path}")

    plt.close()


def plot_all_modes(
    model_name: str,
    dataset_name: str,
    mask_type: str,
    results_dir: str,
    plots_dir: str,
    save_as_pdf: bool = False,
    hop_type: str = "merged",
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 8))

    total_samples = 0

    for mode, style in MODE_STYLES.items():
        try:
            df = load_audio_masking_results(model_name, results_dir, dataset_name, mask_type, mode)
            if dataset_name.startswith("sakura-"):
                df = filter_by_hop_type(df, hop_type)
            if df.empty:
                continue

            consistency_curve = df.groupby("mask_percent")["consistency_value"].mean() * 100
            consistency_curve.sort_index(inplace=True)

            ax.plot(
                consistency_curve.index,
                consistency_curve.values,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=f"{style['label']} (Consistency)",
                linewidth=2,
                markersize=8,
            )

            if "is_correct" in df.columns:
                accuracy_curve = df.groupby("mask_percent")["is_correct"].mean() * 100
                accuracy_curve.sort_index(inplace=True)
                ax.plot(
                    accuracy_curve.index,
                    accuracy_curve.values,
                    marker=style["marker"],
                    linestyle="--",
                    color=style["color"],
                    label=f"{style['label']} (Accuracy)",
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.6,
                )

            total_samples = max(total_samples, len(df["id"].unique()))
            missing = int(df.attrs.get("missing_baseline_count", 0))
            if missing > 0:
                print(f"  - WARNING: {missing}/{len(df)} rows missing baseline join for mode '{mode}'.")
        except FileNotFoundError:
            print(f"  - No data for mode '{mode}'. Skipping.")

    ax.set_title(
        "Partial Audio Masking "
        f"({mask_type.title()}) - All Modes Comparison\n"
        f"{model_name.upper()} on {dataset_name.upper()} ({total_samples} samples)",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("% of Audio Masked", fontsize=12)
    ax.set_ylabel("Consistency with Baseline (%)", fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.legend(title="Mask Mode", loc="best")
    fig.tight_layout()

    output_dir = os.path.join(plots_dir, model_name, "partial_audio_masking", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    base_filename = f"partial_audio_masking_{model_name}_{dataset_name}_{mask_type}_all_modes"

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved: {png_path}")

    if save_as_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format="pdf")
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
    hop_type: str = "merged",
):
    mask_types = ["silence", "noise"] if mask_type == "all" else [mask_type]

    if hop_type == "all":
        hop_runs = ["single", "multi"]
    else:
        hop_runs = [hop_type]

    for ht in hop_runs:
        effective_plots_dir = os.path.join(plots_dir, f"hop_{ht}") if ht != "merged" else plots_dir

        for mt in mask_types:
            hop_label = f" / hop={ht}" if ht != "merged" else ""
            print(
                f"\n--- Generating Partial Audio Masking Plot: "
                f"{model_name.upper()} / {dataset_name} / {mt}{hop_label} ---"
            )

            if mask_mode == "all":
                plot_all_modes(model_name, dataset_name, mt, results_dir, effective_plots_dir, save_as_pdf, ht)
            else:
                try:
                    df = load_audio_masking_results(model_name, results_dir, dataset_name, mt, mask_mode)
                    if dataset_name.startswith("sakura-"):
                        df = filter_by_hop_type(df, ht)
                    missing = int(df.attrs.get("missing_baseline_count", 0))
                    if missing > 0:
                        print(f"  - WARNING: {missing}/{len(df)} rows missing baseline join.")
                    print(f"  - Source file: {df.attrs.get('source_path', 'unknown')}")
                    plot_single_graph(df, model_name, dataset_name, mt, mask_mode, effective_plots_dir, save_as_pdf, ht)
                except FileNotFoundError as err:
                    print(f"  - {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Partial Audio Masking plots.")
    parser.add_argument("--model", type=str, required=True, help="Model name (qwen, salmonn, flamingo)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or 'all'")
    parser.add_argument("--mask-type", type=str, required=True, choices=["silence", "noise", "all"])
    parser.add_argument("--mask-mode", type=str, required=True, choices=["scattered", "start", "end", "all"])
    parser.add_argument(
        "--hop-type",
        type=str,
        default="merged",
        choices=["merged", "single", "multi", "all"],
        help=(
            "Hop type filter for Sakura datasets.\n"
            "  merged = all data together (default)\n"
            "  single = single-hop only\n"
            "  multi  = multi-hop only\n"
            "  all    = run both single and multi separately"
        ),
    )
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--plots_dir", type=str, default="./plots")
    parser.add_argument("--save-pdf", action="store_true")

    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["mmar", "sakura-animal", "sakura-emotion", "sakura-gender", "sakura-language"]
        for dataset in datasets:
            create_analysis(
                args.model,
                dataset,
                args.mask_type,
                args.mask_mode,
                args.results_dir,
                args.plots_dir,
                args.save_pdf,
                args.hop_type,
            )
    else:
        create_analysis(
            args.model,
            args.dataset,
            args.mask_type,
            args.mask_mode,
            args.results_dir,
            args.plots_dir,
            args.save_pdf,
            args.hop_type,
        )
