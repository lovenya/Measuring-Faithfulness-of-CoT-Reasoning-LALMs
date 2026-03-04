# analysis/utils.py

import os
import json
import pandas as pd

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
    # Construct the model-specific path, e.g., 'results/qwen_omni/baseline/'
    experiment_path = os.path.join(results_dir, model_name, experiment_name)
    
    # --- FILENAME CONSTRUCTION ---
    # e.g., 'baseline_qwen_omni_mmar'
    base_name = f"{experiment_name}_{model_name}_{dataset_name}"
    
    # Append -restricted suffix (comes right after dataset name, before other suffixes)
    if restricted:
        base_name += "-restricted"
    
    # Append suffix for lorem filler type
    if filler_type == 'lorem':
        base_name += "-lorem"
    
    # Append suffix for Mistral perturbation source
    if perturbation_source == 'mistral':
        base_name += "-mistral"
    
    filename = f"{base_name}.jsonl"
    # --- END OF FILENAME CONSTRUCTION ---
    
    full_path = os.path.join(experiment_path, filename)

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


def discover_datasets(model_name: str, results_dir: str) -> list:
    """
    Discovers available datasets by scanning the baseline directory for combined JSONL files.
    Excludes .part_ files.

    Returns:
        list: Sorted list of dataset names (e.g., ['mmar', 'sakura-animal', ...]).
    """
    baseline_dir = os.path.join(results_dir, model_name, 'baseline')
    return sorted(list(set([
        f.replace(f'baseline_{model_name}_', '').replace('.jsonl', '')
        for f in os.listdir(baseline_dir)
        if f.endswith('.jsonl') and '.part_' not in f and '-restricted' not in f
    ])))


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