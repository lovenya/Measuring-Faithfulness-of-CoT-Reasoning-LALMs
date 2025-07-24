# analysis/utils.py
import pandas as pd
import os
from typing import List

def load_results(results_dir: str, experiment_name: str, dataset_name: str) -> pd.DataFrame:
    """
    Loads the results from a specified experiment's JSONL file into a pandas DataFrame.
    
    Args:
        results_dir (str): The root directory for results (e.g., './results').
        experiment_name (str): The name of the experiment (e.g., 'baseline').
        dataset_name (str): The short name of the dataset (e.g., 'mmar').
        
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    file_path = os.path.join(results_dir, experiment_name, f"{experiment_name}_{dataset_name}.jsonl")
    if not os.path.exists(file_path):
        print(f"FATAL: Results file not found at '{file_path}'.")
        raise FileNotFoundError(f"Required data file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    return pd.read_json(file_path, lines=True)