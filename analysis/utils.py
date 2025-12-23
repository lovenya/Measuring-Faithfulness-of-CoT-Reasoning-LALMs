# analysis/utils.py

import os
import json
import pandas as pd

def load_results(model_name: str, results_dir: str, experiment_name: str, dataset_name: str, is_restricted: bool, filler_type: str = 'dots', perturbation_source: str = 'self') -> pd.DataFrame:
    """
    Loads experiment results from a model-specific JSONL file into a Pandas DataFrame.

    This function is the single source of truth for constructing file paths. It now
    correctly handles the distinction between 'full' and 'restricted' datasets,
    filler types, and perturbation sources.

    Args:
        model_name (str): The name of the model (e.g., 'qwen', 'salmonn').
        results_dir (str): The root directory for all results (e.g., './results').
        experiment_name (str): The name of the experiment (e.g., 'baseline').
        dataset_name (str): The short name of the dataset (e.g., 'mmar').
        is_restricted (bool): If True, loads the '-restricted.jsonl' version of the file.
        filler_type (str): Type of filler used (e.g., 'dots', 'lorem'). Defaults to 'dots'.
        perturbation_source (str): Source of perturbations ('self' or 'mistral'). Defaults to 'self'.

    Raises:
        FileNotFoundError: If the specified results file does not exist.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded results.
    """
    # Construct the model-specific path, e.g., 'results/qwen/baseline/'
    experiment_path = os.path.join(results_dir, model_name, experiment_name)
    
    # --- FILENAME CONSTRUCTION ---
    # Build filename with appropriate suffixes based on flags
    if is_restricted:
        # e.g., 'baseline_qwen_mmar-restricted'
        base_name = f"{experiment_name}_{model_name}_{dataset_name}-restricted"
    else:
        # e.g., 'baseline_qwen_mmar'
        base_name = f"{experiment_name}_{model_name}_{dataset_name}"
    
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
        # Use a list comprehension for efficient line-by-line reading of the JSONL file.
        data = [json.loads(line) for line in open(full_path, 'r')]
        
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