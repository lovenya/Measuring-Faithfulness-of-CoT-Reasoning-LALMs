#!/usr/bin/env python3
"""
Test script for Coqui TTS XTTS-v2 local inference
Works with locally downloaded models on compute clusters without internet
"""

import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils import io
import time

# Monkey patch the load_fsspec function to use weights_only=False
original_load_fsspec = io.load_fsspec

def patched_load_fsspec(path, map_location=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load_fsspec(path, map_location, **kwargs)

io.load_fsspec = patched_load_fsspec

def test_xtts_inference():
    """Test XTTS-v2 model inference with sample texts"""
    
    # Model paths
    model_path = "/project/rrg-csubakan/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/tts_models/XTTS-v2"
    config_path = os.path.join(model_path, "config.json")
    model_file = os.path.join(model_path, "model.pth")
    vocab_file = os.path.join(model_path, "vocab.json")
    speakers_file = os.path.join(model_path, "speakers_xtts.pth")
    
    # Check if required files exist
    required_files = [config_path, model_file, vocab_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            return False
    
    # Check if speakers file exists (may be optional depending on model version)
    speakers_file_exists = os.path.exists(speakers_file)
    print(f"Speakers file exists: {speakers_file_exists}")
    
    print("All required files found. Loading model...")
    
    # Load configuration
    config = XttsConfig()
    config.load_json(config_path)
    
    # Initialize model
    model = Xtts.init_from_config(config)
    
    # Load model weights
    print("Loading model weights...")
    model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Test texts
    test_texts = [
        "Hello, this is a test of the XTTS-v2 text to speech model running locally.",
        "The model appears to be working correctly on the compute cluster environment."
    ]
    
    # Reference speaker audio file
    speaker_wav = "/project/rrg-csubakan/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/tts_models/reference_speaker.wav"
    
    # Check if reference audio exists
    if not os.path.exists(speaker_wav):
        print(f"ERROR: Reference audio file not found: {speaker_wav}")
        return False
    
    print(f"Using reference speaker: {speaker_wav}")
    
    # Language setting
    language = "en"  # English
    
    # Create output directory
    output_dir = "tts_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each test text
    for i, text in enumerate(test_texts):
        print(f"\nProcessing text {i+1}: {text[:50]}...")
        
        try:
            start_time = time.time()
            
            # Generate speech with reference speaker
            outputs = model.synthesize(
                text,
                config,
                speaker_wav=speaker_wav,
                gpt_cond_len=3,
                language=language,
                enable_text_splitting=True
            )
            
            inference_time = time.time() - start_time
            
            # Debug: Check what we got back
            print(f"  Output type: {type(outputs)}")
            if isinstance(outputs, dict):
                print(f"  Output keys: {outputs.keys()}")
                for key, value in outputs.items():
                    print(f"    {key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
            elif hasattr(outputs, 'shape'):
                print(f"  Output shape: {outputs.shape}")
            else:
                print(f"  Output: {outputs}")
            
            # Save output
            output_file = os.path.join(output_dir, f"test_output_{i+1}.wav")
            
            # Handle different output formats
            if isinstance(outputs, dict) and 'wav' in outputs:
                audio = outputs['wav']
            elif isinstance(outputs, torch.Tensor):
                audio = outputs
            else:
                audio = outputs  # Assume it's already audio data
            
            print(f"  Audio type: {type(audio)}")
            if hasattr(audio, 'shape'):
                print(f"  Audio shape: {audio.shape}")
                print(f"  Audio min/max: {audio.min():.4f}/{audio.max():.4f}")
            
            # Convert to tensor and ensure correct format
            if isinstance(audio, torch.Tensor):
                audio_tensor = audio
            elif hasattr(audio, '__array__'):  # numpy array or similar
                import numpy as np
                audio_tensor = torch.from_numpy(np.array(audio, dtype=np.float32))
                print(f"  Converted numpy to tensor")
            else:
                print(f"  ERROR: Audio format not supported: {type(audio)}")
                continue
            
            # Ensure correct dimensions
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            
            print(f"  Final audio tensor shape: {audio_tensor.shape}")
            print(f"  Saving to: {output_file}")
            
            # Save audio
            try:
                torchaudio.save(output_file, audio_tensor.cpu(), sample_rate=22050)
                print(f"  torchaudio.save completed")
                
                # Verify file was created
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"  File created successfully, size: {file_size} bytes")
                else:
                    print(f"  ERROR: File not created!")
                    
            except Exception as save_error:
                print(f"  ERROR saving audio: {save_error}")
                continue
            
            print(f"✓ Generated audio saved to: {output_file}")
            print(f"  Inference time: {inference_time:.2f} seconds")
            print(f"  Audio duration: {audio.shape[-1] / 22050:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Error processing text {i+1}: {str(e)}")
            continue
    
    print(f"\nTest completed! Check the '{output_dir}' directory for generated audio files.")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import TTS
        print(f"✓ TTS library version: {TTS.__version__}")
    except ImportError:
        print("✗ TTS library not found. Install with: pip install TTS")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import torchaudio
        print(f"✓ Torchaudio version: {torchaudio.__version__}")
    except ImportError:
        print("✗ Torchaudio not found")
        return False
    
    return True

if __name__ == "__main__":
    print("XTTS-v2 Local Inference Test")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        exit(1)
    
    # Run inference test
    success = test_xtts_inference()
    
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed. Check error messages above.")