# Ultra-conservative download for resource-limited login nodes
import os
import time
from huggingface_hub import hf_hub_download, HfApi

# Configuration
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
CACHE_DIR = '/lustre09/project/6090520/lovenya/.cache'

# Disable all fast/parallel downloads
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Reduce output spam

def download_with_delays():
    """Download files one by one with delays to be nice to login node"""
    
    # Essential files first (small, quick downloads)
    essential_files = [
        "config.json",
        "generation_config.json", 
        "tokenizer_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json"
    ]
    
    print("=== Phase 1: Downloading config/tokenizer files ===")
    for filename in essential_files:
        try:
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=filename,
                cache_dir=CACHE_DIR
            )
            print(f"  ✓ {filename}")
            time.sleep(1)  # Be nice to the system
        except Exception as e:
            print(f"  ⚠ {filename}: {e}")
    
    print("\n=== Phase 2: Downloading model weights ===")
    try:
        # Get the large model files
        api = HfApi()
        repo_files = api.list_repo_files(MODEL_ID)
        model_files = [f for f in repo_files if f.endswith('.safetensors') and f != 'model.safetensors.index.json']
        
        for i, filename in enumerate(model_files, 1):
            print(f"Downloading model file {i}/{len(model_files)}: {filename}")
            try:
                hf_hub_download(
                    repo_id=MODEL_ID,
                    filename=filename,
                    cache_dir=CACHE_DIR
                )
                print(f"  ✓ {filename}")
                time.sleep(2)  # Longer delay for large files
            except Exception as e:
                print(f"  ✗ Failed: {filename}: {e}")
                print("  You may need to try this file again later")
        
        print(f"\nDownload complete! Model should be cached at:")
        print(f"   {CACHE_DIR}")
        
    except Exception as e:
        print(f"Error listing files: {e}")

if __name__ == "__main__":
    print("Starting ultra-conservative download...")
    print("This will be slow but gentle on login node resources")
    download_with_delays()
    
    
    
    

# Next commands will be:
# ls/lustre09/project/6090520/lovenya/.cache/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots
# Copy the hash
# cp -RL /lustre09/project/6090520/lovenya/.cache/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/<that_hash>/* ./Qwen2-Audio-7B-Instruct/


