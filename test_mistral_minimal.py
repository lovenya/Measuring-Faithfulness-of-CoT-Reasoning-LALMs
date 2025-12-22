#!/usr/bin/env python3
"""
Minimal Mistral test - simple approach without extra abstraction.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Point to local offline path
model_path = "/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/model_components/mistral-small-3"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # Critical for H100
    device_map="auto"            # Automatically puts it on the GPU
)

# Prepare Input
messages = [
    {"role": "user", "content": "What is 2 + 2? Answer briefly."}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

# Generate
print("Generating...")
outputs = model.generate(
    inputs, 
    max_new_tokens=50, 
    temperature=0.1,
    do_sample=True
)

# Decode
print("Response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("\nDone!")
