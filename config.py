# config.py

# --- Path Configurations ---
MODEL_PATH = "./Qwen2-Audio-7B-Instruct" 
RESULTS_DIR = "./results"

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
    
    # --- NEW: 'transcribed_audio' condition datasets ---
    "mmar-transcribed_audio": "data/mmar_transcribed/mmar_transcribed_audio_standardized.jsonl",
    "sakura-animal-transcribed_audio": "data/sakura_transcribed/animal/sakura_animal_transcribed_audio_standardized.jsonl",
    "sakura-emotion-transcribed_audio": "data/sakura_transcribed/emotion/sakura_emotion_transcribed_audio_standardized.jsonl",
    "sakura-gender-transcribed_audio": "data/sakura_transcribed/gender/sakura_gender_transcribed_audio_standardized.jsonl",
    "sakura-language-transcribed_audio": "data/sakura_transcribed/language/sakura_language_transcribed_audio_standardized.jsonl",
}

# --- Experiment Default Parameters ---
NUM_SAMPLES_TO_RUN = 0 
NUM_CHAINS_PER_QUESTION = 10
SNR_LEVELS_TO_TEST = [20, 10, 5, 0, -5, -10]

# --- Global Variables (Managed by main.py) ---
DATASET_NAME = "default"
BASELINE_RESULTS_FILE_OVERRIDE = None
VERBOSE = False
OUTPUT_PATH = None
CONDITION = "default" # The new global flag for the experimental condition