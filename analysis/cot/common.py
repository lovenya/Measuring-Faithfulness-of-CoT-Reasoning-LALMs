"""Shared utilities for CoT analysis scripts."""

from __future__ import annotations

import json
import os

import pandas as pd


def _filler_suffix(filler_type: str) -> str:
    return f"-{filler_type}"

def default_reference_mode(experiment: str) -> str:
    defaults = {
        "paraphrasing": "0pct",
        "early_answering": "100pct",
        "partial_filler_text": "0pct",
        "adding_mistakes": "baseline",
    }
    if experiment not in defaults:
        raise ValueError(f"Unsupported experiment: {experiment}")
    return defaults[experiment]


def discover_datasets(model_name: str, results_dir: str, restricted: bool = False) -> list[str]:
    baseline_dir = os.path.join(results_dir, model_name, "baseline")
    prefix = f"baseline_{model_name}_"
    datasets = set()

    for filename in os.listdir(baseline_dir):
        if not filename.endswith(".jsonl"):
            continue
        if ".part_" in filename:
            continue
        if not filename.startswith(prefix):
            continue

        dataset_part = filename[len(prefix) : -len(".jsonl")]
        has_restricted_suffix = dataset_part.endswith("-restricted")

        if restricted:
            if not has_restricted_suffix:
                continue
            dataset_part = dataset_part[: -len("-restricted")]
        else:
            if has_restricted_suffix:
                continue

        datasets.add(dataset_part)

    return sorted(datasets)


def _build_results_path(
    model_name: str,
    results_dir: str,
    experiment_name: str,
    dataset_name: str,
    filler_type: str,
    perturbation_source: str,
    partial_filler_mode: str,
    restricted: bool,
) -> str:
    if experiment_name == "partial_filler_text":
        filename_base = f"{experiment_name}_{model_name}_{dataset_name}"
        if restricted:
            filename_base += "-restricted"
        filename_base += _filler_suffix(filler_type)
        filename = f"{filename_base}_{partial_filler_mode}.jsonl"
        return os.path.join(results_dir, model_name, experiment_name, partial_filler_mode, filename)

    experiment_path = os.path.join(results_dir, model_name, experiment_name)
    filename_base = f"{experiment_name}_{model_name}_{dataset_name}"

    if restricted:
        filename_base += "-restricted"

    if "filler" in experiment_name:
        filename_base += _filler_suffix(filler_type)

    if perturbation_source != "self":
        filename_base += f"-{perturbation_source}"

    return os.path.join(experiment_path, f"{filename_base}.jsonl")


def load_results(
    model_name: str,
    results_dir: str,
    experiment_name: str,
    dataset_name: str,
    filler_type: str = "dots",
    perturbation_source: str = "self",
    partial_filler_mode: str = "start",
    restricted: bool = False,
) -> pd.DataFrame:
    path = _build_results_path(
        model_name=model_name,
        results_dir=results_dir,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        filler_type=filler_type,
        perturbation_source=perturbation_source,
        partial_filler_mode=partial_filler_mode,
        restricted=restricted,
    )

    if experiment_name == "partial_filler_text" and not os.path.exists(path):
        legacy_canonical_base = f"{experiment_name}_{model_name}_{dataset_name}"
        if restricted:
            legacy_canonical_base += "-restricted"
        legacy_canonical_path = os.path.join(
            results_dir,
            model_name,
            experiment_name,
            partial_filler_mode,
            f"{legacy_canonical_base}_{partial_filler_mode}.jsonl",
        )
        if os.path.exists(legacy_canonical_path):
            path = legacy_canonical_path

    if experiment_name == "partial_filler_text" and not os.path.exists(path):
        # Legacy fallback support for old split experiment names.
        legacy_name_by_mode = {
            "start": "partial_filler_text",
            "end": "flipped_partial_filler_text",
            "random": "random_partial_filler_text",
        }
        legacy_exp = legacy_name_by_mode[partial_filler_mode]
        legacy_base = f"{legacy_exp}_{model_name}_{dataset_name}"
        if restricted:
            legacy_base += "-restricted"
        candidate_legacy_bases = [f"{legacy_base}-{filler_type}"]
        if filler_type == "dots":
            candidate_legacy_bases.append(legacy_base)
        for candidate_base in candidate_legacy_bases:
            legacy_path = os.path.join(
                results_dir,
                model_name,
                legacy_exp,
                f"{candidate_base}.jsonl",
            )
            if os.path.exists(legacy_path):
                path = legacy_path
                break

    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(rows)


def _compute_progress_percentage(df: pd.DataFrame, experiment: str) -> pd.Series:
    if experiment == "paraphrasing":
        valid = df["total_sentences_in_chain"] > 0
        result = pd.Series(float("nan"), index=df.index)
        result[valid] = (
            df.loc[valid, "num_sentences_paraphrased"]
            / df.loc[valid, "total_sentences_in_chain"]
            * 100.0
        )
        return result

    if experiment == "early_answering":
        valid = df["total_sentences_in_chain"] > 0
        result = pd.Series(float("nan"), index=df.index)
        result[valid] = (
            df.loc[valid, "num_sentences_provided"]
            / df.loc[valid, "total_sentences_in_chain"]
            * 100.0
        )
        return result

    if experiment == "partial_filler_text":
        return df["percent_replaced"].astype(float)

    if experiment == "adding_mistakes":
        valid = df["total_sentences_in_chain"] > 0
        result = pd.Series(float("nan"), index=df.index)
        result[valid] = (
            (df.loc[valid, "mistake_position"] - 1)
            / df.loc[valid, "total_sentences_in_chain"]
            * 100.0
        )
        return result

    raise ValueError(f"Unsupported experiment: {experiment}")


def prepare_experiment_frame(df: pd.DataFrame, experiment: str, bin_size: int) -> pd.DataFrame:
    frame = df.copy()
    frame["progress_pct"] = _compute_progress_percentage(frame, experiment)
    frame = frame[frame["progress_pct"].notna()].copy()
    frame["progress_binned"] = (frame["progress_pct"] / bin_size).round() * bin_size
    frame["id"] = frame["id"].astype(str)
    frame["chain_id"] = frame["chain_id"].astype(str)
    return frame


def build_reference_lookup(
    experiment_df: pd.DataFrame,
    reference_mode: str,
    baseline_df: pd.DataFrame | None = None,
) -> dict[tuple[str, str], str | None]:
    if reference_mode == "baseline":
        if baseline_df is None:
            raise ValueError("baseline_df is required when reference_mode='baseline'")
        working = baseline_df[["id", "chain_id", "predicted_choice"]].copy()
        working["id"] = working["id"].astype(str)
        working["chain_id"] = working["chain_id"].astype(str)
        return dict(zip(zip(working["id"], working["chain_id"]), working["predicted_choice"]))

    if reference_mode == "0pct":
        zero_df = experiment_df[experiment_df["progress_pct"] == 0][["id", "chain_id", "predicted_choice"]].copy()
        return dict(zip(zip(zero_df["id"], zero_df["chain_id"]), zero_df["predicted_choice"]))

    if reference_mode == "100pct":
        max_rows = experiment_df.groupby(["id", "chain_id"], as_index=False)["progress_pct"].max()
        max_rows = max_rows.rename(columns={"progress_pct": "max_progress"})
        merged = experiment_df.merge(max_rows, on=["id", "chain_id"], how="inner")
        merged = merged[merged["progress_pct"] == merged["max_progress"]]
        merged = merged.drop_duplicates(subset=["id", "chain_id"], keep="last")
        merged = merged[["id", "chain_id", "predicted_choice"]]
        return dict(zip(zip(merged["id"], merged["chain_id"]), merged["predicted_choice"]))

    raise ValueError(f"Unsupported reference_mode: {reference_mode}")


def reference_label(reference_mode: str) -> str:
    labels = {
        "baseline": "baseline",
        "0pct": "0% condition",
        "100pct": "100% condition",
    }
    return labels[reference_mode]


def x_axis_label(experiment: str) -> str:
    labels = {
        "paraphrasing": "Percentage (%) of sentences paraphrased",
        "early_answering": "Percentage (%) of reasoning chain provided",
        "partial_filler_text": "Percentage (%) of words replaced",
        "adding_mistakes": "Percentage (%) of chain before inserted mistake",
    }
    return labels[experiment]
