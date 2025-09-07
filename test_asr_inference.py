# #!/usr/bin/env python3
# """
# Test script for Whisper Large v3 inference on Compute Canada cluster
# Usage: python test_whisper_inference.py
# """

# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import librosa
# import time
# import os
# import sys

# def check_environment():
#     """Check if environment is set up correctly"""
#     print("=== Environment Check ===")
#     print(f"Python version: {sys.version}")
#     print(f"PyTorch version: {torch.__version__}")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"CUDA devices: {torch.cuda.device_count()}")
#         print(f"Current device: {torch.cuda.current_device()}")
#         print(f"Device name: {torch.cuda.get_device_name()}")
#         print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
#     print()

# def load_model(model_path):
#     """Load Whisper model from local path"""
#     print(f"Loading model from: {model_path}")
    
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model path not found: {model_path}")
    
#     start_time = time.time()
    
#     # Load processor components separately (more robust)
#     print("Loading processor components...")
#     try:
#         from transformers import WhisperFeatureExtractor, WhisperTokenizer
#         feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
#         tokenizer = WhisperTokenizer.from_pretrained(model_path, local_files_only=True)
        
#         # Create processor manually
#         from transformers import WhisperProcessor
#         processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#         print("Processor created successfully using separate components")
#     except Exception as e:
#         print(f"Failed to load processor components separately: {e}")
#         print("Trying direct processor loading...")
#         processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
    
#     # Load model with GPU optimization
#     print("Loading model...")
#     model = WhisperForConditionalGeneration.from_pretrained(
#         model_path, 
#         local_files_only=True,
#         torch_dtype=torch.float16,  # Use half precision for memory efficiency
#         device_map="auto",         # Automatically map to available GPU
#         low_cpu_mem_usage=True     # Reduce CPU memory usage during loading
#     )
    
#     load_time = time.time() - start_time
#     print(f"Model loaded in {load_time:.2f} seconds")
#     print(f"Model device: {model.device}")
    
#     if torch.cuda.is_available():
#         print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
#     return processor, model

# def transcribe_audio(audio_path, processor, model, language=None):
#     """Transcribe audio file"""
#     print(f"\n=== Transcribing Audio ===")
#     print(f"Audio file: {audio_path}")
    
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
#     # Load and preprocess audio
#     print("Loading audio file...")
#     start_time = time.time()
#     audio, sr = librosa.load(audio_path, sr=16000)
#     load_audio_time = time.time() - start_time
    
#     print(f"Audio loaded in {load_audio_time:.2f} seconds")
#     print(f"Audio duration: {len(audio) / 16000:.2f} seconds")
#     print(f"Sample rate: {sr} Hz")
    
#     # Process audio
#     print("Processing audio features...")
#     process_start = time.time()
#     input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
#     input_features = input_features.to(model.device)
#     process_time = time.time() - process_start
#     print(f"Audio processing took {process_time:.2f} seconds")
    
#     # Generate transcription
#     print("Generating transcription...")
#     inference_start = time.time()
    
#     with torch.no_grad():  # Disable gradient computation for inference
#         if language:
#             print(f"Using forced language: {language}")
#             forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
#             predicted_ids = model.generate(
#                 input_features, 
#                 forced_decoder_ids=forced_decoder_ids,
#                 max_length=448,  # Reasonable max length
#                 num_beams=1,     # Faster inference with beam search disabled
#                 do_sample=False  # Deterministic output
#             )
#         else:
#             print("Auto-detecting language...")
#             predicted_ids = model.generate(
#                 input_features,
#                 max_length=448,
#                 num_beams=1,
#                 do_sample=False
#             )
    
#     inference_time = time.time() - inference_start
    
#     # Decode transcription
#     print("Decoding transcription...")
#     decode_start = time.time()
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#     decode_time = time.time() - decode_start
    
#     total_time = time.time() - start_time
    
#     print(f"\n=== Timing Results ===")
#     print(f"Audio loading: {load_audio_time:.2f}s")
#     print(f"Audio processing: {process_time:.2f}s")
#     print(f"Inference: {inference_time:.2f}s")
#     print(f"Decoding: {decode_time:.2f}s")
#     print(f"Total transcription time: {total_time:.2f}s")
    
#     if torch.cuda.is_available():
#         print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
#     return transcription

# def main():
#     # Configuration
#     MODEL_PATH = os.path.expanduser("~/asr_models/whisper-large-v3")
#     AUDIO_PATH = "data/mmar/audio/mmar_audio_3.wav"
#     LANGUAGE = None  # Set to "en" for English, "fr" for French, etc. or None for auto-detect
    
#     print("=== Whisper Large v3 Test Inference ===\n")
    
#     try:
#         # Check environment
#         check_environment()
        
#         # Load model
#         processor, model = load_model(MODEL_PATH)
        
#         # Test transcription
#         transcription = transcribe_audio(AUDIO_PATH, processor, model, LANGUAGE)
        
#         # Display results
#         print(f"\n=== TRANSCRIPTION RESULT ===")
#         print(f"'{transcription}'")
        
#         # Save to file
#         output_file = "transcription_output.txt"
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(f"Audio file: {AUDIO_PATH}\n")
#             f.write(f"Model: Whisper Large v3\n")
#             f.write(f"Language: {LANGUAGE if LANGUAGE else 'Auto-detect'}\n")
#             f.write(f"Transcription: {transcription}\n")
        
#         print(f"\nTranscription saved to: {output_file}")
        
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
    
#     print("\n=== Test completed successfully! ===")

# if __name__ == "__main__":
#     main()