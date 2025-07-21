# --- Path Configurations ---
# Assumes the model is in the project root. Update if it's elsewhere.
MODEL_PATH = "./Qwen2-Audio-7B-Instruct" 
RESULTS_DIR = "./results"

# --- Experiment Default Parameters ---
# Set to 0 to run on the entire dataset by default.
# This can be overridden with the --num-samples flag for testing.
NUM_SAMPLES_TO_RUN = 0 

# UPDATED: Default number of reasoning chains to generate for each question.
# This can be overridden with the --num-chains flag for testing.
NUM_CHAINS_PER_QUESTION = 10

# --- Global Variables (Managed by main.py) ---
# These are placeholders that will be dynamically set by the orchestrator.
# Do not change them here.
DATASET_NAME = "default"
BASELINE_RESULTS_FILE_OVERRIDE = None