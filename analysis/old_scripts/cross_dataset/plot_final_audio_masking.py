# analysis/cross_dataset/plot_final_audio_masking.py

"""Generate cross-dataset plots for the partial audio masking experiment.

Consistency is computed at analysis time by joining experiment outputs with
baseline predictions on (id, chain_id).
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_STYLES = {
    "mmar": {"label": "MMAR", "color": "#e41a1c", "marker": "X"},
    "sakura-animal": {"label": "S.Animal", "color": "#377eb8", "marker": "o"},
    "sakura-emotion": {"label": "S.Emotion", "color": "#4daf4a", "marker": "v"},
    "sakura-gender": {"label": "S.Gender", "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "S.Language", "color": "#984ea3", "marker": ">"},
}

MODE_LINESTYLES = {
    "scattered": "-",
    "start": "--",
    "end": ":",
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


def filter_df_by_hop_type(df: pd.DataFrame, hop_type: str) -> pd.DataFrame:
    if hop_type == "merged" or df.empty:
        return df
    mask = df.apply(lambda row: get_hop_type(row.to_dict()) == hop_type, axis=1)
    return df[mask].reset_index(drop=True)


def _candidate_modes(mask_mode: str) -> list[str]:
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

    candidate_files.append(
        os.path.join(results_dir, model_name, "audio_masking", f"audio_masking_{model_name}_{dataset_name}.jsonl")
    )

    for filepath in candidate_files:
        if not os.path.exists(filepath):
            continue

        rows = _load_jsonl(filepath)
        df = pd.DataFrame(rows)

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
    hop_type: str = "merged",
):
    hop_label = f" (hop={hop_type})" if hop_type != "merged" else ""
    print(f"\n--- Cross-Dataset Plot: {model_name.upper()} / {mask_type} / {mask_mode}{hop_label} ---")

    fontsize = 32
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    all_dfs = []

    for dataset_name, style in DATASET_STYLES.items():
        try:
            df = load_audio_masking_results(model_name, results_dir, dataset_name, mask_type, mask_mode)

            if df.empty:
                print(f"  - No data for {dataset_name}. Skipping.")
                continue

            df["dataset"] = dataset_name
            if dataset_name.startswith("sakura-"):
                df = filter_df_by_hop_type(df, hop_type)
            if df.empty:
                print(f"  - No data for {dataset_name} after hop_type filter. Skipping.")
                continue

            missing = int(df.attrs.get("missing_baseline_count", 0))
            if missing > 0:
                print(f"  - WARNING: {dataset_name} has {missing}/{len(df)} rows missing baseline joins.")
            print(f"  - Source ({dataset_name}): {df.attrs.get('source_path', 'unknown')}")

            df["consistency_pct"] = df["consistency_value"] * 100
            all_dfs.append(df)

            sns.lineplot(
                data=df,
                x="mask_percent",
                y="consistency_pct",
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=MODE_LINESTYLES.get(mask_mode, "-"),
                linewidth=2,
                markersize=20,
                errorbar=("ci", 95) if show_ci else None,
                ax=ax,
                legend=False,
            )

            if print_line_data:
                curve = df.groupby("mask_percent")["consistency_value"].mean() * 100
                print(f"  {dataset_name}: X={curve.index.tolist()}, Y={[round(y, 2) for y in curve.values]}")

        except FileNotFoundError:
            print(f"  - No data for {dataset_name}. Skipping.")

    if not all_dfs:
        print("  - No data found for any dataset. Aborting plot.")
        plt.close()
        return

    title = f"Partial Audio Masking ({mask_type.title()}, {mask_mode.title()}), {model_name.upper()}"
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("% of Audio Masked", fontsize=fontsize)
    ax.set_ylabel("Consistency (%)", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=(fontsize - 4))

    if y_zoom:
        ax.set_ylim(y_zoom[0], y_zoom[1])
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)
    ax.grid(True)
    fig.tight_layout()

    output_dir = os.path.join(plots_dir, model_name, "partial_audio_masking")
    os.makedirs(output_dir, exist_ok=True)

    base_filename = f"cross_dataset_partial_audio_masking_{model_name}_{mask_type}_{mask_mode}"

    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300)
    print(f"  - Plot saved: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, format="pdf")
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
    hop_type: str = "merged",
):
    mask_types = ["silence", "noise"] if mask_type == "all" else [mask_type]
    mask_modes = ["scattered", "start", "end"] if mask_mode == "all" else [mask_mode]

    if hop_type == "all":
        hop_runs = ["single", "multi"]
    else:
        hop_runs = [hop_type]

    for ht in hop_runs:
        effective_plots_dir = os.path.join(plots_dir, f"hop_{ht}") if ht != "merged" else plots_dir

        for mt in mask_types:
            for mm in mask_modes:
                create_cross_dataset_plot(
                    model_name,
                    mt,
                    mm,
                    results_dir,
                    effective_plots_dir,
                    y_zoom,
                    save_pdf,
                    show_ci,
                    print_line_data,
                    ht,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cross-dataset Partial Audio Masking plots.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Model name (qwen, salmonn, flamingo)")
    parser.add_argument("--mask-type", type=str, required=True, choices=["silence", "noise", "all"])
    parser.add_argument("--mask-mode", type=str, required=True, choices=["scattered", "start", "end", "all"])
    parser.add_argument(
        "--hop-type",
        type=str,
        default="merged",
        choices=["merged", "single", "multi", "all"],
        help="Hop type filter for Sakura (merged|single|multi|all)",
    )
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--plots_dir", type=str, default="plots/cross_dataset_plots")
    parser.add_argument("--y-zoom", nargs=2, type=float, default=None, help="Custom Y-axis range")
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--show-ci", action="store_true", help="Show 95%% confidence interval")
    parser.add_argument("--print-line-data", action="store_true", help="Print line coordinates to console")

    args = parser.parse_args()

    create_analysis(
        args.model,
        args.mask_type,
        args.mask_mode,
        args.results_dir,
        args.plots_dir,
        args.y_zoom,
        args.save_pdf,
        args.show_ci,
        args.print_line_data,
        args.hop_type,
    )
