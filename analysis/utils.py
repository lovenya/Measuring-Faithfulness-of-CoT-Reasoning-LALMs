# analysis/utils.py

import pandas as pd
import os
from typing import Optional

def get_results_path(results_dir: str, experiment_name: str, dataset_name: str, condition: str) -> str:
    """
    Constructs the correct, condition-aware path to a results file.
    This is the single source of truth for all file path logic in the analysis scripts.
    """
    # --- THIS IS THE CORRECTED, FINAL LOGIC ---
    if condition == 'default':
        # Default condition files are in the top-level experiment folder.
        top_level_dir = os.path.join(results_dir, experiment_name)
        filename = f"{experiment_name}_{dataset_name}.jsonl"
    else:
        # All other conditions are in a dedicated subdirectory.
        condition_dir = f"{condition}_experiments"
        top_level_dir = os.path.join(results_dir, condition_dir, experiment_name)
        filename = f"{experiment_name}_{dataset_name}_{condition}.jsonl"
    # --- END OF CORRECTION ---
    
    # Construct the full path.
    full_path = os.path.join(top_level_dir, filename)
    return full_path

def load_results(results_dir: str, experiment_name: str, dataset_name: str, condition: str) -> pd.DataFrame:
    """
    Loads results from a specified experiment's JSONL file into a pandas DataFrame
    using the definitive path construction logic.
    """
    # Get the one, true path from our helper function.
    file_path = get_results_path(results_dir, experiment_name, dataset_name, condition)
    
    if not os.path.exists(file_path):
        print(f"FATAL: Results file not found.")
        print(f"  - Experiment: {experiment_name}")
        print(f"  - Dataset:    {dataset_name}")
        print(f"  - Condition:  {condition}")
        print(f"  - Looked for: {file_path}")
        raise FileNotFoundError(f"Required data file not found at {file_path}")
    
    print(f"Loading data from: {file_path}")
    try:
        return pd.read_json(file_path, lines=True)
    except ValueError as e:
        print(f"ERROR: Could not parse JSONL file: {file_path}")
        print(f"This often happens if the file is empty or corrupted.")
        print(f"Error details: {e}")
        # Return an empty DataFrame to allow the analysis to skip this dataset gracefully.
        return pd.DataFrame()