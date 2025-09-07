# debug_tts_load.py

import os
import torch
import time

print(f"--- Starting TTS Load Debug Script ---")
print(f"Timestamp: {time.ctime()}")

try:
    print("Importing TTS library...")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils import io
    print("✓ TTS library imported successfully.")
except Exception as e:
    print(f"✗ FAILED to import TTS library: {e}")
    exit(1)

# --- Monkey Patch ---
original_load_fsspec = io.load_fsspec
def patched_load_fsspec(path, map_location=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load_fsspec(path, map_location, **kwargs)
io.load_fsspec = patched_load_fsspec
print("✓ Monkey patch applied.")

def setup_tts_model(model_dir: str):
    """
    A direct copy of the setup function with maximum logging.
    """
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] --- Entering setup_tts_model function ---")
    
    config_path = os.path.join(model_dir, "config.json")
    print(f"[{time.time() - start_time:.2f}s] Config path set to: {config_path}")

    print(f"[{time.time() - start_time:.2f}s] Loading model configuration from JSON...")
    config = XttsConfig()
    config.load_json(config_path)
    print(f"[{time.time() - start_time:.2f}s] ✓ Configuration loaded.")
    
    print(f"[{time.time() - start_time:.2f}s] Initializing model from configuration...")
    model = Xtts.init_from_config(config)
    print(f"[{time.time() - start_time:.2f}s] ✓ Model initialized.")
    
    print(f"[{time.time() - start_time:.2f}s] Loading model weights from checkpoint... (This is the suspected hang point)")
    model.load_checkpoint(config, checkpoint_dir=model_dir, use_deepspeed=False)
    print(f"[{time.time() - start_time:.2f}s] ✓ Model weights loaded.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.time() - start_time:.2f}s] Moving TTS model to device: {device}...")
    model.to(device)
    print(f"[{time.time() - start_time:.2f}s] ✓ Model moved to GPU.")
    
    print(f"[{time.time() - start_time:.2f}s] --- TTS Model setup complete. ---")
    return model

if __name__ == "__main__":
    tts_model_dir = './tts_models/XTTS-v2'
    print(f"Attempting to load model from: {tts_model_dir}")
    
    try:
        setup_tts_model(tts_model_dir)
        print("\n--- SCRIPT COMPLETED SUCCESSFULLY ---")
    except Exception as e:
        print(f"\n--- SCRIPT FAILED WITH AN EXCEPTION ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
