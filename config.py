# --- Path Configurations ---
# Assumes the model is in the project root. Update if it's elsewhere.
MODEL_PATH = "./Qwen2-Audio-7B-Instruct" 
RESULTS_DIR = "./results"

# --- Dataset Alias Mapping ---
DATASET_MAPPING = {
    # --- Original Clean Datasets ---
    "mmar": "data/mmar/mmar_test_standardized.jsonl",
    "sakura-animal": "data/sakura/animal/sakura_animal_test_standardized.jsonl",
    "sakura-emotion": "data/sakura/emotion/sakura_emotion_standardized.jsonl",
    "sakura-gender": "data/sakura/gender/sakura_gender_standardized.jsonl",
    "sakura-language": "data/sakura/language/sakura_language_standardized.jsonl",

    # --- NEW: Noisy Datasets ---
    "mmar-noisy": "data/mmar_noisy/mmar_noisy_standardized.jsonl",
    "sakura-animal-noisy": "data/sakura_noisy/animal/animal_noisy_standardized.jsonl",
    "sakura-emotion-noisy": "data/sakura_noisy/emotion/emotion_noisy_standardized.jsonl",
    "sakura-gender-noisy": "data/sakura_noisy/gender/gender_noisy_standardized.jsonl",
    "sakura-language-noisy": "data/sakura_noisy/language/language_noisy_standardized.jsonl",
}

# --- Experiment Default Parameters ---
# Set to 0 to run on the entire dataset by default.
# This can be overridden with the --num-samples flag for testing.
NUM_SAMPLES_TO_RUN = 0 

# UPDATED: Default number of reasoning chains to generate for each question.
# This can be overridden with the --num-chains flag for testing.
NUM_CHAINS_PER_QUESTION = 10

# SNR Levels for Noise Robustness Experiment
SNR_LEVELS_TO_TEST = [20, 10, 5, 0, -5, -10]

# --- Global Variables (Managed by main.py) ---
# These are placeholders that will be dynamically set by the orchestrator.
# Do not change them here.
DATASET_NAME = "default"
BASELINE_RESULTS_FILE_OVERRIDE = None
VERBOSE = False
OUTPUT_PATH = None