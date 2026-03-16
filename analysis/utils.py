# analysis/utils.py

import os
import json
import pandas as pd

FILLER_EXPERIMENTS = {
    "filler_text",
    "partial_filler_text",
    "random_partial_filler_text",
    "flipped_partial_filler_text",
}

PARTIAL_FILLER_MODE_BY_EXPERIMENT = {
    "partial_filler_text": "start",
    "flipped_partial_filler_text": "end",
    "random_partial_filler_text": "random",
}

INVALID_BASELINE_SUFFIXES = {
    "dots",
    "lorem",
    "mistral",
}


def _is_clean_baseline_dataset_name(dataset_part: str) -> bool:
    """Reject polluted baseline filenames like 'mmar-lorem'."""
    return not any(dataset_part.endswith(f"-{suffix}") for suffix in INVALID_BASELINE_SUFFIXES)

def load_results(model_name: str, results_dir: str, experiment_name: str, dataset_name: str, filler_type: str = 'dots', perturbation_source: str = 'self', restricted: bool = False) -> pd.DataFrame:
    """
    Loads experiment results from a model-specific JSONL file into a Pandas DataFrame.

    This function is the single source of truth for constructing file paths.

    Args:
        model_name (str): The name of the model (e.g., 'qwen_omni', 'flamingo_hf').
        results_dir (str): The root directory for all results (e.g., './results').
        experiment_name (str): The name of the experiment (e.g., 'baseline').
        dataset_name (str): The short name of the dataset (e.g., 'mmar').
        filler_type (str): Type of filler used (e.g., 'dots', 'lorem'). Defaults to 'dots'.
        perturbation_source (str): Source of perturbations ('self' or 'mistral'). Defaults to 'self'.
        restricted (bool): If True, load the restricted version (-restricted suffix). Defaults to False.

    Raises:
        FileNotFoundError: If the specified results file does not exist.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded results.
    """
    partial_filler_mode = PARTIAL_FILLER_MODE_BY_EXPERIMENT.get(experiment_name)
    canonical_experiment_name = (
        "partial_filler_text" if partial_filler_mode is not None else experiment_name
    )

    if canonical_experiment_name == "partial_filler_text":
        experiment_path = os.path.join(
            results_dir,
            model_name,
            canonical_experiment_name,
            partial_filler_mode,
        )
    else:
        # Construct the model-specific path, e.g., 'results/qwen_omni/baseline/'
        experiment_path = os.path.join(results_dir, model_name, canonical_experiment_name)
    
    # --- FILENAME CONSTRUCTION ---
    # e.g., 'baseline_qwen_omni_mmar'
    base_name = f"{canonical_experiment_name}_{model_name}_{dataset_name}"
    
    # Append -restricted suffix (comes right after dataset name, before other suffixes)
    if restricted:
        base_name += "-restricted"
    
    candidate_base_names = [base_name]

    if experiment_name in FILLER_EXPERIMENTS:
        if canonical_experiment_name == "partial_filler_text":
            candidate_base_names = [f"{base_name}-{filler_type}_{partial_filler_mode}"]
            if filler_type == "dots":
                # Backward compatibility for older unsuffixed dots outputs.
                candidate_base_names.append(f"{base_name}_{partial_filler_mode}")
        else:
            candidate_base_names = [f"{base_name}-{filler_type}"]
            if filler_type == "dots":
                # Backward compatibility for older unsuffixed dots outputs.
                candidate_base_names.append(base_name)

    # Append suffix for Mistral perturbation source
    if perturbation_source == 'mistral':
        candidate_base_names = [f"{name}-mistral" for name in candidate_base_names]

    candidate_paths = [
        os.path.join(experiment_path, f"{name}.jsonl")
        for name in candidate_base_names
    ]

    if partial_filler_mode is not None and experiment_name != "partial_filler_text":
        legacy_base_name = f"{experiment_name}_{model_name}_{dataset_name}"
        if restricted:
            legacy_base_name += "-restricted"
        legacy_candidate_base_names = [f"{legacy_base_name}-{filler_type}"]
        if filler_type == "dots":
            legacy_candidate_base_names.append(legacy_base_name)
        if perturbation_source == 'mistral':
            legacy_candidate_base_names = [f"{name}-mistral" for name in legacy_candidate_base_names]
        candidate_paths.extend(
            os.path.join(results_dir, model_name, experiment_name, f"{name}.jsonl")
            for name in legacy_candidate_base_names
        )
    # --- END OF FILENAME CONSTRUCTION ---

    full_path = next((path for path in candidate_paths if os.path.exists(path)), candidate_paths[0])

    try:
        # Read line-by-line, skipping any corrupted lines gracefully.
        data = []
        skipped = 0
        with open(full_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
                    continue
        
        if skipped > 0:
            print(f"  - WARNING: Skipped {skipped} corrupted line(s) in {full_path}")
        
        if not data:
            # Handle the case of an empty results file.
            print(f"  - WARNING: Results file is empty: {full_path}")
            return pd.DataFrame()

        return pd.DataFrame(data)

    except FileNotFoundError:
        # Provide a clear, actionable error message if a required file is missing.
        print(f"\nFATAL ERROR: Could not find required results file.")
        print(f"  - Searched for: {full_path}")
        # Re-raise the exception to halt the calling script, preventing partial analysis.
        raise


def discover_datasets(model_name: str, results_dir: str, restricted: bool = False) -> list:
    """
    Discover available datasets by scanning baseline results.

    Args:
        model_name: Model alias (e.g., 'qwen_omni').
        results_dir: Root results directory.
        restricted: If True, discover datasets from files ending in
            '-restricted.jsonl'. If False, discover non-restricted files.

    Returns:
        Sorted dataset aliases (without '-restricted' suffix).
    """
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

        if not _is_clean_baseline_dataset_name(dataset_part):
            continue

        datasets.add(dataset_part)

    return sorted(datasets)


def check_completeness(model_name: str, results_dir: str, experiment_name: str, dataset_name: str, 
                        baseline_df: pd.DataFrame, experiment_df: pd.DataFrame) -> dict:
    """
    Checks if experiment results are complete by comparing unique (id, chain_id) pairs
    against the baseline.

    Args:
        model_name: Model name.
        results_dir: Results root directory.
        experiment_name: Experiment name.
        dataset_name: Dataset name.
        baseline_df: DataFrame with baseline results.
        experiment_df: DataFrame with experiment results.

    Returns:
        dict with keys: 'baseline_count', 'experiment_count', 'pct_complete', 'is_complete'.
    """
    baseline_pairs = set(zip(baseline_df['id'], baseline_df['chain_id']))
    experiment_pairs = set(zip(experiment_df['id'], experiment_df['chain_id']))
    
    baseline_count = len(baseline_pairs)
    experiment_count = len(experiment_pairs & baseline_pairs)
    pct = (experiment_count / baseline_count * 100) if baseline_count > 0 else 0
    
    return {
        'baseline_count': baseline_count,
        'experiment_count': experiment_count,
        'pct_complete': pct,
        'is_complete': experiment_count >= baseline_count
    }
