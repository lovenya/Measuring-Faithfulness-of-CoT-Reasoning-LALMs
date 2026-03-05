"""
Paper-ready shared plot style for cross-dataset analysis figures.

This mirrors the collaborator-provided style contract:
- Very large fonts and thick lines/markers
- Fixed dataset colors/markers
- Legends saved as standalone PDFs (not drawn inside plots)
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. PAPER-READY PLOT SETTINGS (ULTRA-MASSIVE FONTS)
# ==========================================
plt.rcParams.update(
    {
        "font.size": 38,
        "axes.labelsize": 46,
        "axes.titlesize": 50,
        "xtick.labelsize": 38,
        "ytick.labelsize": 38,
        "legend.fontsize": 34,
        "legend.title_fontsize": 38,
        "lines.linewidth": 6,
        "lines.markersize": 24,
        "figure.autolayout": True,
    }
)

# Canonical dataset order used in this repo's naming convention.
DATASET_ORDER = [
    "sakura-animal",
    "sakura-language",
    "sakura-emotion",
    "sakura-gender",
    "mmar",
    "mmau",
]

# Keep collaborator style values exactly.
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "p"]
LINE_STYLES = ["-", "--", ":", "-."]

DATASET_LABELS = {
    "sakura-animal": "ANIMAL",
    "sakura-language": "LANGUAGE",
    "sakura-emotion": "EMOTION",
    "sakura-gender": "GENDER",
    "mmar": "MMAR",
    "mmau": "MMAU",
}


def ordered_present_datasets(dataset_names: Iterable[str]) -> list[str]:
    """
    Return present dataset names in canonical plot order.
    Unknown dataset names are appended alphabetically at the end.
    """
    present = set(dataset_names)
    ordered = [d for d in DATASET_ORDER if d in present]
    extras = sorted([d for d in present if d not in DATASET_ORDER])
    return ordered + extras


def dataset_style(dataset_name: str) -> dict[str, str]:
    """Get color/marker/label style for a dataset key."""
    if dataset_name in DATASET_ORDER:
        idx = DATASET_ORDER.index(dataset_name)
        return {
            "color": COLORS[idx],
            "marker": MARKERS[idx],
            "label": DATASET_LABELS.get(dataset_name, dataset_name.upper()),
        }
    # Fallback style for any unexpected dataset.
    return {"color": "#333333", "marker": "o", "label": dataset_name.upper()}


def generate_dataset_legend(output_dir: str, datasets_in_plot: Iterable[str] | None = None) -> str:
    """
    Create a standalone dataset legend PDF (no in-plot legend).
    Returns path to saved legend.
    """
    os.makedirs(output_dir, exist_ok=True)
    if datasets_in_plot is None:
        datasets = DATASET_ORDER
    else:
        datasets = ordered_present_datasets(datasets_in_plot)

    with plt.rc_context({"figure.autolayout": False}):
        fig, ax = plt.subplots(figsize=(22, 2))
        ax.axis("off")

        handles = []
        labels = []
        for dataset in datasets:
            style = dataset_style(dataset)
            line = Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=8,
                marker=style["marker"],
                markersize=26,
            )
            handles.append(line)
            labels.append(style["label"])

        fig.legend(
            handles,
            labels,
            loc="center",
            ncol=len(handles),
            fontsize=38,
            frameon=False,
            handletextpad=0.8,
            columnspacing=1.5,
        )

        legend_path = os.path.join(output_dir, "legend_datasets_only.pdf")
        plt.savefig(legend_path, format="pdf", bbox_inches="tight")
        plt.close()
        return legend_path


def generate_model_legend(models: list[str], output_dir: str) -> str:
    """
    Create a standalone model legend PDF using line styles.
    Included for compatibility with the reference style module.
    """
    os.makedirs(output_dir, exist_ok=True)
    with plt.rc_context({"figure.autolayout": False}):
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")

        handles = []
        labels = []
        for i, model_name in enumerate(models):
            line = Line2D(
                [0],
                [0],
                color="black",
                linewidth=8,
                linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            )
            handles.append(line)
            labels.append(model_name.upper())

        fig.legend(
            handles,
            labels,
            loc="center",
            ncol=len(handles),
            fontsize=38,
            frameon=False,
            handletextpad=0.8,
            columnspacing=1.5,
        )

        legend_path = os.path.join(output_dir, "legend_models_only.pdf")
        plt.savefig(legend_path, format="pdf", bbox_inches="tight")
        plt.close()
        return legend_path
