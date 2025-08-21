# analysis/utils.py
import pandas as pd
import os
from typing import List, Optional

def load_results(results_dir: str, experiment_name: str, dataset_name: str, condition: Optional[str] = None) -> pd.DataFrame:
    """
    Loads results from a specified experiment's JSONL file into a pandas DataFrame.
    This version is now condition-aware and backward-compatible.
    """
    # --- THE CRITICAL FIX ---
    # This logic now correctly handles our file naming history.
    if condition:
        # For new conditions, construct the specific filename.
        filename = f"{experiment_name}_{dataset_name}_{condition}.jsonl"
        file_path = os.path.join(results_dir, experiment_name, filename)
        
        # This is the key change: if the condition is 'default', we must also
        # check for the old filename format for backward compatibility.
        if condition == 'default' and not os.path.exists(file_path):
            print(f"  - INFO: Could not find '{filename}'. Falling back to original filename format.")
            old_filename = f"{experiment_name}_{dataset_name}.jsonl"
            file_path = os.path.join(results_dir, experiment_name, old_filename)
    else:
        # If no condition is specified, use the original filename format.
        filename = f"{experiment_name}_{dataset_name}.jsonl"
        file_path = os.path.join(results_dir, experiment_name, filename)
    # --- END OF FIX ---
    
    if not os.path.exists(file_path):
        print(f"FATAL: Results file not found. Checked for: '{filename}'")
        if condition == 'default':
            print(f"       Also checked for: '{old_filename}'")
        raise FileNotFoundError(f"Required data file not found.")
    
    print(f"Loading data from: {file_path}")
    return pd.read_json(file_path, lines=True)