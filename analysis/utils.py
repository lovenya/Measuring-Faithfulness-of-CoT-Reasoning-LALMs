# analysis/utils.py
import pandas as pd
import os
from typing import List, Optional

def load_results(results_dir: str, experiment_name: str, dataset_name: str, condition: Optional[str] = None) -> pd.DataFrame:
    """
    Loads results from a specified experiment's JSONL file into a pandas DataFrame.
    This version is now condition-aware.

    Args:
        results_dir (str): The root directory for results (e.g., './results').
        experiment_name (str): The name of the experiment (e.g., 'baseline').
        dataset_name (str): The short name of the dataset (e.g., 'mmar').
        condition (Optional[str]): The experimental condition (e.g., 'default', 'transcribed_audio').
                                   If None, assumes the old filename format.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    # --- THE CRITICAL FIX ---
    # Construct the filename based on whether a condition is provided.
    if condition:
        # New, condition-aware filename format
        filename = f"{experiment_name}_{dataset_name}_{condition}.jsonl"
    else:
        # Old, original filename format for backward compatibility
        filename = f"{experiment_name}_{dataset_name}.jsonl"
    # --- END OF FIX ---

    file_path = os.path.join(results_dir, experiment_name, filename)
    
    if not os.path.exists(file_path):
        print(f"FATAL: Results file not found at '{file_path}'.")
        raise FileNotFoundError(f"Required data file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    return pd.read_json(file_path, lines=True)