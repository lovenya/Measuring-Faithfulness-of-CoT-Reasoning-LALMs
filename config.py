# config.py

"""
This file is the central configuration hub for the entire research framework.
It contains all the master paths, model aliases, and default experimental
parameters. This single source of truth makes our framework robust and easy to modify.
"""

# --- Path Configurations ---
# This dictionary holds the master paths to all our model assets.
MODEL_PATHS = {
    # --- Qwen (Self-Contained) ---
    "qwen": "/scratch/lovenya/models/Qwen/Qwen2-Audio-7B",

    # --- Audio Flamingo (Complex) ---
    "flamingo_code": "./audio-flamingo-code",
    "flamingo_weights": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/audio-flamingo-weights",

    # --- SALMONN (Multi-Component) ---
    "salmonn_checkpoint": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/salmonn-13b-checkpoint/salmonn_v1.pth",
    "salmonn_7b_checkpoint": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/salmonn-7b-checkpoint/salmonn_7b_v0.pth",

    # --- Mistral Small 3 (External Perturbation Model) ---
    # Used for generating mistakes and paraphrasing to avoid in-distribution bias
    "mistral_small_3": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/mistral-small-3",
}

# Check if local SSD model directory is available (set by submission script)
import os
_LOCAL_MODEL_DIR = os.environ.get('SALMONN_LOCAL_MODEL_DIR', None)

if _LOCAL_MODEL_DIR and os.path.exists(_LOCAL_MODEL_DIR):
    # Use local NVMe SSD paths for faster loading
    SALMONN_COMPONENT_PATHS = {
        "source_code": "./salmonn-source-code",
        "whisper": os.path.join(_LOCAL_MODEL_DIR, "whisper-large-v2"),
        "beats": os.path.join(_LOCAL_MODEL_DIR, "beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
        "vicuna": os.path.join(_LOCAL_MODEL_DIR, "vicuna-13b-v1.1"),
        "salmonn_checkpoint": os.path.join(_LOCAL_MODEL_DIR, "salmonn-13b-checkpoint/salmonn_v1.pth"),
        "bert_base": os.path.join(_LOCAL_MODEL_DIR, "bert-base-uncased"),
    }
    
    SALMONN_7B_COMPONENT_PATHS = {
        "source_code": "./salmonn-source-code",
        "whisper": os.path.join(_LOCAL_MODEL_DIR, "whisper-large-v2"),
        "beats": os.path.join(_LOCAL_MODEL_DIR, "beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
        "vicuna": os.path.join(_LOCAL_MODEL_DIR, "vicuna-7b-v1.5"),
        "salmonn_checkpoint": os.path.join(_LOCAL_MODEL_DIR, "salmonn-7b-checkpoint/salmonn_7b_v0.pth"),
        "bert_base": os.path.join(_LOCAL_MODEL_DIR, "bert-base-uncased"),
    }
    print(f"[CONFIG] Using LOCAL SSD model paths from: {_LOCAL_MODEL_DIR}")
else:
    # Default: use network /scratch paths
    SALMONN_COMPONENT_PATHS = {
        "source_code": "./salmonn-source-code",
        "whisper": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/whisper-large-v2",
        "beats": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        "vicuna": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/vicuna-13b-v1.1",
        "salmonn_checkpoint": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/salmonn-13b-checkpoint/salmonn_v1.pth",
        "bert_base": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/bert-base-uncased",
    }

    SALMONN_7B_COMPONENT_PATHS = {
        "source_code": "./salmonn-source-code",
        "whisper": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/whisper-large-v2",
        "beats": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/beats_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        "vicuna": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/vicuna-7b-v1.5",
        "salmonn_checkpoint": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/salmonn-7b-checkpoint/salmonn_7b_v0.pth",
        "bert_base": "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/bert-base-uncased",
    }

# The main directory where all experimental results will be saved.
RESULTS_DIR = "./results"

# A crucial asset for our 'silent audio' methodology for text-only inference.
SILENT_AUDIO_PATH = "./assets/silent.wav"


# --- Model Aliases ---
# This maps the short, user-friendly names from the command line
# to the internal keys used in MODEL_PATHS.
MODEL_ALIASES = {
    "qwen": "qwen",
    "flamingo": "flamingo_weights",
    # This line tells main.py: "when the user types 'salmonn', the key you
    # need to look for in MODEL_PATHS is 'salmonn_checkpoint'".
    "salmonn": "salmonn_checkpoint",
    "salmonn_7b": "salmonn_7b_checkpoint",
}


# --- Dataset Alias Mapping ---
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

    "jasco": "data/jasco/jasco_standardized.jsonl",
}

MMAR_DATASET_PATH = "data/mmar"
SAKURA_DATASET_PATH = "data/sakura"

# --- Experiment Default Parameters ---
NUM_SAMPLES_TO_RUN = 0
NUM_CHAINS_PER_QUESTION = 1
SNR_LEVELS_TO_TEST = [20, 10, 5, 0, -5, -10]
FILLER_TYPE = "dots"  # Options: "dots", "lorem"


# --- Global Variables (Managed by main.py) ---
MODEL_ALIAS = "default"
DATASET_NAME = "default"
BASELINE_RESULTS_FILE_OVERRIDE = None
VERBOSE = True
OUTPUT_PATH = None