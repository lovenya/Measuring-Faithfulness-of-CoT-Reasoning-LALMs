# config.py

# --- Path Configurations ---
# This dictionary holds the master paths to all our model assets.
MODEL_PATHS = {
    # --- Qwen (Self-Contained) ---
    "qwen": "./Qwen2-Audio-7B-Instruct",

    # --- Audio Flamingo (Complex) ---
    "flamingo_code": "./audio-flamingo-code",
    "flamingo_weights": "./audio-flamingo-weights",

    # --- NEW: SALMONN (Multi-Component) ---
    "salmonn_code": "./salmonn-source-code",
    "salmonn_whisper": "./model_components/whisper-large-v2",
    "salmonn_beats": "./model_components/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    "salmonn_vicuna": "./model_components/vicuna-13b-v1.1",
    "salmonn_checkpoint": "./model_components/salmonn-13b-checkpoint",
    
}

# This is the main directory where all experimental results will be saved.
RESULTS_DIR = "./results"

# This is a crucial asset for our 'silent audio' methodology for text-only inference.
SILENT_AUDIO_PATH = "./assets/silent.wav"


# --- Model Aliases ---
# This maps the short, user-friendly names from the command line
# to the internal keys used in MODEL_PATHS.
MODEL_ALIASES = {
    "qwen": "qwen",
    "flamingo": "flamingo_weights",
    # The 'salmonn' alias points to its main checkpoint directory, which is the
    # primary path our utility script needs to locate the final .pth file.
    "salmonn": "salmonn_checkpoint"
}


# --- Dataset Alias Mapping ---
# This is the single source of truth for all dataset paths.
DATASET_MAPPING = {
    "mmar": "data/mmar/mmar_test_standardized.jsonl",
    "sakura-animal": "data/sakura/animal/sakura_animal_test_standardized.jsonl",
    "sakura-emotion": "data/sakura/emotion/sakura_emotion_test_standardized.jsonl",
    "sakura-gender": "data/sakura/gender/sakura_gender_test_standardized.jsonl",
    "sakura-language": "data/sakura/language/sakura_language_test_standardized.jsonl",

    "mmar-noisy": "data/mmar_noisy/mmar_noisy_standardized.jsonl",
    "sakura-animal-noisy": "data/sakura_noisy/animal/animal_noisy_standardized.jsonl",
    "sakura-emotion-noisy": "data/sakura_noisy/emotion/emotion_noisy_standardized.jsonl",
    "sakura-gender-noisy": "data/sakura_noisy/gender/gender_noisy_standardized.jsonl",
    "sakura-language-noisy": "data/sakura_noisy/language/language_noisy_standardized.jsonl",
}


# --- Experiment Default Parameters ---
NUM_SAMPLES_TO_RUN = 0
NUM_CHAINS_PER_QUESTION = 10
SNR_LEVELS_TO_TEST = [20, 10, 5, 0, -5, -10]


# --- Global Variables (Managed by main.py) ---
MODEL_ALIAS = "default"
DATASET_NAME = "default"
BASELINE_RESULTS_FILE_OVERRIDE = None
VERBOSE = True
OUTPUT_PATH = None