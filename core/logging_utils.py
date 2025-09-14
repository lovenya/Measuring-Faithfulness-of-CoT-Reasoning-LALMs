# core/logging_utils.py

import logging
import os

def setup_logger(log_dir: str, model_alias: str, experiment_name: str, dataset_name: str):
    """
    Sets up a robust, file-based logger for a specific experiment run.
    """
    # Create a unique, informative log filename.
    log_filename = f"{experiment_name}_{model_alias}_{dataset_name}.log"
    log_filepath = os.path.join(log_dir, model_alias, experiment_name, log_filename)
    
    # Ensure the directory for the log file exists.
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    # Define the standard format for our log messages.
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format)

    # Get the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level to log.

    # --- File Handler ---
    # This handler writes all logs (INFO and above) to our dedicated log file.
    file_handler = logging.FileHandler(log_filepath, mode='a') # Append mode
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- Console Handler ---
    # This handler prints logs to the console (stdout).
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logger initialized successfully.")
    return log_filepath