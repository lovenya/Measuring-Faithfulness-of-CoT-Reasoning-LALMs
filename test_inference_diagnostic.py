# test_inference_job.py - Diagnostic Version

import torch
import librosa
import json
import os
import sys
from pathlib import Path

# Add diagnostic imports
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("ERROR: transformers not installed")
    sys.exit(1)

try:
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    print("Successfully imported Qwen2AudioForConditionalGeneration")
except ImportError as e:
    print(f"ERROR: Could not import Qwen2AudioForConditionalGeneration: {e}")
    # Try alternative import
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        print("Using AutoModelForCausalLM as fallback")
        Qwen2AudioForConditionalGeneration = AutoModelForCausalLM
    except ImportError:
        print("ERROR: Could not import any suitable model class")
        sys.exit(1)

# --- Configuration: Define Paths ---
LOCAL_MODEL_PATH = "./Qwen2-Audio-7B-Instruct" 
MMAR_JSONL_PATH = "./data/mmar/mmar_test_standardized.jsonl"
SAKURA_EMOTION_JSONL_PATH = "./data/sakura/emotion/sakura_emotion_test_standardized.jsonl"

# --- Function to diagnose model directory ---
def diagnose_model_directory(model_path: str):
    """Diagnose the model directory structure and files."""
    print(f"\n--- Diagnosing Model Directory: {model_path} ---")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model directory {model_path} does not exist!")
        return False
    
    print(f"Model directory exists: {model_path}")
    
    # Check for required files
    required_files = [
        "config.json",
        "pytorch_model.bin",  # or model.safetensors
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    optional_files = [
        "model.safetensors",
        "generation_config.json",
        "preprocessor_config.json"
    ]
    
    print("\nChecking for required files:")
    for file in required_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        print(f"  {file}: {'✓' if exists else '✗'}")
        if not exists:
            # Check for alternative names
            if file == "pytorch_model.bin":
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    print(f"    Found model.safetensors instead")
    
    print("\nChecking for optional files:")
    for file in optional_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        print(f"  {file}: {'✓' if exists else '✗'}")
    
    # Check config.json content
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"\nModel architecture: {config.get('model_type', 'Unknown')}")
            print(f"Torch dtype: {config.get('torch_dtype', 'Unknown')}")
            print(f"Architectures: {config.get('architectures', 'Unknown')}")
        except Exception as e:
            print(f"ERROR reading config.json: {e}")
    
    return True

# --- Function to Load Model and Processor with Better Error Handling ---
def load_qwen_audio_model_and_processor(model_path: str):
    """Loads the Qwen-Audio model and its associated processor with detailed error handling."""
    print(f"\n--- Loading Model and Processor ---")
    
    # First diagnose the directory
    if not diagnose_model_directory(model_path):
        raise Exception("Model directory diagnosis failed")
    
    print(f"\nAttempting to load model from {model_path}...")
    
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            print(f"Using device: {device}")
        else:
            print("CUDA not available, using CPU")
            device = "cpu"
        
        # Try loading processor first (usually fails first if there's an issue)
        print("Step 1: Loading processor...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_path, 
                use_fast=False,
                trust_remote_code=True
            )
            print("✓ Processor loaded successfully")
        except Exception as e:
            print(f"✗ Processor loading failed: {e}")
            # Try without trust_remote_code
            print("Trying without trust_remote_code...")
            try:
                processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
                print("✓ Processor loaded successfully (without trust_remote_code)")
            except Exception as e2:
                print(f"✗ Processor loading failed completely: {e2}")
                raise e2
        
        # Try loading model
        print("Step 2: Loading model...")
        try:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            # Try without trust_remote_code
            print("Trying without trust_remote_code...")
            try:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto" if torch.cuda.is_available() else None
                )
                print("✓ Model loaded successfully (without trust_remote_code)")
            except Exception as e2:
                print(f"✗ Model loading failed completely: {e2}")
                # Try with low_cpu_mem_usage
                print("Trying with low_cpu_mem_usage=True...")
                try:
                    model = Qwen2AudioForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype="auto",
                        device_map="auto" if torch.cuda.is_available() else None,
                        low_cpu_mem_usage=True
                    )
                    print("✓ Model loaded successfully (with low_cpu_mem_usage)")
                except Exception as e3:
                    print(f"✗ All model loading attempts failed: {e3}")
                    raise e3
        
        print(f"Model device: {next(model.parameters()).device}")
        return model, processor
        
    except Exception as e:
        print(f"ERROR: Failed to load model or processor from {model_path}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        
        # Additional debugging information
        print("\nDebugging information:")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
        
        # List all files in model directory
        print(f"\nAll files in {model_path}:")
        try:
            for file in os.listdir(model_path):
                file_path = os.path.join(model_path, file)
                size = os.path.getsize(file_path) if os.path.isfile(file_path) else "DIR"
                print(f"  {file}: {size}")
        except Exception as list_error:
            print(f"Could not list directory: {list_error}")
        
        raise

# --- Simplified test function ---
def test_model_loading():
    """Test just the model loading without full inference."""
    print("--- Testing Model Loading Only ---")
    
    try:
        model, processor = load_qwen_audio_model_and_processor(LOCAL_MODEL_PATH)
        print("\n✓ SUCCESS: Model and processor loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Processor type: {type(processor).__name__}")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: Could not load model: {e}")
        return False

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Starting Diagnostic Test ---")
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test model loading
    success = test_model_loading()
    
    if success:
        print("\n--- Model Loading Test PASSED ---")
        print("You can now run the full inference script.")
    else:
        print("\n--- Model Loading Test FAILED ---")
        print("Please check the error messages above and fix the issues.")
        sys.exit(1)
    
    print("\n--- Diagnostic Test Finished ---")