# config.py

# --- Path Configurations ---
# This dictionary holds the paths to the root directories of our models.
# The keys are internal identifiers that we'll use throughout the project.
MODEL_PATHS = {
    "qwen": "./Qwen2-Audio-7B-Instruct",
    
    # --- Paths for Audio Flamingo ---
    # We now have two distinct paths: one for the weights and one for the code.
    
    "flamingo_code": "./audio-flamingo-code",
    "flamingo_weights": "./audio-flamingo-weights",
    
}

# This is the main directory where all experimental results will be saved.
# Our main.py script will create model-specific subdirectories inside this folder.
RESULTS_DIR = "./results"


# --- Model Aliases ---
# This maps the short, user-friendly names you'll use on the command line
# (e.g., --model qwen) to the internal keys used in MODEL_PATHS.
# This makes our commands cleaner and allows us to change paths without
# breaking the command-line interface.
MODEL_ALIASES = {
    "qwen": "qwen",
    "flamingo": "flamingo_weights"
}

# --- Dataset Alias Mapping ---
# This is the single source of truth for all dataset paths.
# The keys are designed to be constructed from the command-line arguments.
DATASET_MAPPING = {
    # --- 'default' condition datasets ---
    "mmar": "data/mmar/mmar_test_standardized.jsonl",
    "sakura-animal": "data/sakura/animal/sakura_animal_test_standardized.jsonl",
    "sakura-emotion": "data/sakura/emotion/sakura_emotion_standardized.jsonl",
    "sakura-gender": "data/sakura/gender/sakura_gender_standardized.jsonl",
    "sakura-language": "data/sakura/language/sakura_language_standardized.jsonl",

    # --- 'noisy' condition datasets (for robustness_to_noise experiment) ---
    "mmar-noisy": "data/mmar_noisy/mmar_noisy_standardized.jsonl",
    "sakura-animal-noisy": "data/sakura_noisy/animal/animal_noisy_standardized.jsonl",
    "sakura-emotion-noisy": "data/sakura_noisy/emotion/emotion_noisy_standardized.jsonl",
    "sakura-gender-noisy": "data/sakura_noisy/gender/gender_noisy_standardized.jsonl",
    "sakura-language-noisy": "data/sakura_noisy/language/language_noisy_standardized.jsonl",
}

# --- Experiment Default Parameters ---
# These are the default settings for a full, scientific run. They can be
# temporarily overridden from the command line for quick tests.
NUM_SAMPLES_TO_RUN = 0  # Set to 0 to run on the entire dataset by default.
NUM_CHAINS_PER_QUESTION = 5
SNR_LEVELS_TO_TEST = [20, 10, 5, 0, -5, -10]


# --- Global Variables (Managed by main.py) ---
# These are placeholders that our main orchestrator will fill in at runtime.
# This allows any script to know the context of the current run.
MODEL_ALIAS = "default"
DATASET_NAME = "default"
BASELINE_RESULTS_FILE_OVERRIDE = None
VERBOSE = False
OUTPUT_PATH = None