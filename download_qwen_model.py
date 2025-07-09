# download_qwen_model.py
import os
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# This is the Hugging Face Hub ID for the model
model_id = "Qwen/Qwen2-Audio-7B-Instruct"

# Set a specific cache directory if you want, or leave it to default ~/.cache/huggingface
# os.environ['HF_HOME'] = '/project/def-csubakan-ab/lovenya/.cache'

print(f"--- Starting download for: {model_id} ---")
print("This script will trigger the download of the model files to the Hugging Face cache.")
print("It is expected to be KILLED by the system if it uses too much memory, which is NORMAL.")
print("The goal is just to get the files downloaded, not to load the model successfully here.")

# These lines will download the model files to your ~/.cache/huggingface/hub directory
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id)

print("--- This message will likely not be seen. If it is, the download is complete. ---")