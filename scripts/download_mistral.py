#!/usr/bin/env python3
# scripts/download_mistral.py

"""
Downloads Mistral Small 3 weights to the project's model directory.
Run this on the login node to pre-fetch weights.
"""

import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Target directory
TARGET_DIR = "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/mistral-small-3"
MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

def download_model():
    print(f"Downloading {MODEL_ID} to {TARGET_DIR}...")
    
    # Create directory if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(TARGET_DIR)
    
    # Download model
    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
    )
    # We don't need to save_pretrained again if we just want the cache, 
    # but to be safe and have a clean directory structure, let's save it.
    # However, loading large models on login node might kill the process (OOM).
    # BETTER APPROACH: Use snapshot_download which just downloads files without loading into RAM.
    
    return True

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    
    print(f"Downloading {MODEL_ID} to {TARGET_DIR} using snapshot_download...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional cleanup
    )
    print("Download complete!")
