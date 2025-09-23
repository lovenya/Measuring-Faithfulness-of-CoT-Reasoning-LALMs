import json
import random
import librosa
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def load_model(model_path="./Qwen2-Audio-7B-Instruct/"):
    """Load the Qwen2-Audio model and processor"""
    print(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16  # Use float16 for efficiency
    )
    print("Model loaded successfully!")
    return model, processor

def format_choices(choices):
    """Format choices as (a) choice1 (b) choice2 etc."""
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    formatted = []
    for i, choice in enumerate(choices):
        formatted.append(f"({letters[i]}) {choice}")
    return " ".join(formatted)

def create_conversation(question, choices, audio_path):
    """Create conversation format for the model"""
    formatted_choices = format_choices(choices)
    full_question = f"{question} {formatted_choices}"
    
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_path": audio_path},
            {"type": "text", "text": full_question},
        ]},
    ]
    return conversation

def run_inference(model, processor, data_point):
    """Run inference on a single data point"""
    print(f"\nProcessing: {data_point['id']}")
    print(f"Question: {data_point['question']}")
    print(f"Choices: {data_point['choices']}")
    print(f"Audio: {data_point['audio_path']}")
    
    # Create conversation
    conversation = create_conversation(
        data_point['question'], 
        data_point['choices'], 
        data_point['audio_path']
    )
    
    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # Load audio
    audio_data, _ = librosa.load(
        data_point['audio_path'], 
        sr=processor.feature_extractor.sampling_rate
    )
    
    
    
    # build the BatchEncoding…
    inputs = processor(
        text=text,
        audio=[audio_data],
        return_tensors="pt",
        padding=True
    )

    # move *all* tensors to the same device as the model:
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    
    
    # Generate response
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=128)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f"Model Response: {response}")
    print(f"Correct Answer Key: {data_point['answer_key']}")
    
    return response

def main():
    # Load model
    model, processor = load_model()
    
    # Sample data points
    mmar_sample = {
        "id": "mmar_3", 
        "audio_path": "data/mmar/audio/mmar_audio_3.wav", 
        "question": "Is this man drunk?", 
        "choices": ["Drunk", "Not drunk"], 
        "answer_key": 0
    }
    
    sakura_sample = {
        "id": "sakura_animal_33_single", 
        "audio_path": "data/sakura/animal/audio/sakura_animal_audio_33.wav", 
        "question": "Can you select the animal from the provided options that matches the sound?", 
        "choices": ["cow", "cat", "hen", "dog"], 
        "answer_key": 1
    }
    
    # Run inference on both samples
    print("="*50)
    print("MMAR Sample Inference")
    print("="*50)
    mmar_response = run_inference(model, processor, mmar_sample)
    
    print("\n" + "="*50)
    print("Sakura Sample Inference")
    print("="*50)
    sakura_response = run_inference(model, processor, sakura_sample)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"MMAR Response: {mmar_response}")
    print(f"Sakura Response: {sakura_response}")

if __name__ == "__main__":
    main()