# config.py

"""
This file is the central configuration hub for the entire research framework.
It acts as the "Single Source of Truth" for all important paths, model aliases,
and default experiment parameters. This design makes our framework modular,
easy to configure, and simple to maintain.
"""

# --- Path Configurations ---
# This dictionary holds the paths to all model-related directories.
MODEL_PATHS = {
    # Qwen is self-contained.
    "qwen": "./Qwen2-Audio-7B-Instruct",
    
    # Flamingo has separate directories for its code and weights.
    "flamingo_code": "./audio-flamingo-code",
    "flamingo_weights": "./audio-flamingo-weights",

    # SALMONN is a multi-component model requiring paths to each part.
    "salmonn_code": "./salmonn-source-code",
    "whisper_path": "./model_components/whisper-large-v2",
    "beats_path": "./model_components/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    "llama_path": "./model_components/vicuna-7b-v1.5",
    "ckpt": "./model_components/salmonn-7b-v0-checkpoint/SALMONN-7B_v0.pth",
}

# This is the main directory where all experimental results will be saved.
# Our main.py script will create model-specific subdirectories inside this folder.
RESULTS_DIR = "./results"

# This is a crucial asset for our "silent audio" methodology, used for pure
# text-only inference with multi-modal models.
SILENT_AUDIO_PATH = "./assets/silent.wav"


# --- Model Aliases ---
# This dictionary acts as a bridge. It maps the short, user-friendly names you
# use on the command line (e.g., --model salmonn) to the internal keys used in
# MODEL_PATHS that point to the primary model weights/checkpoint.
MODEL_ALIASES = {
    "qwen": "qwen",
    "flamingo": "flamingo_weights",
    "salmonn": "ckpt" # The alias points to the main checkpoint file.
}


# --- Dataset Alias Mapping ---
# This is the single source of truth for all dataset paths. The keys are the
# short names you'll use with the --dataset flag.
DATASET_MAPPING = {
    # --- Standard Datasets ---
    "mmar": "data/mmar/mmar_test_standardized.jsonl",
    "sakura-animal": "data/sakura/animal/sakura_animal_test_standardized.jsonl",
    "sakura-emotion": "data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
    "sakura-gender": "data/sakura/gender/sakura_gender_test_standardized.jsonl",
    "sakura-language": "data/sakura/language/sakura_language_test_standardized.jsonl",

    # --- 'Noisy' Datasets (for the robustness_to_noise experiment) ---
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
NUM_CHAINS_PER_QUESTION = 10
SNR_LEVELS_TO_TEST = [20, 10, 5, 0, -5, -10]


# --- Global Variables (Managed by main.py) ---
# These are placeholders that our main orchestrator will fill in at runtime.
# This allows any script to know the context of the current run.
MODEL_ALIAS = "default"
DATASET_NAME = "default"
BASELINE_RESULTS_FILE_OVERRIDE = None
VERBOSE = False
OUTPUT_PATH = None