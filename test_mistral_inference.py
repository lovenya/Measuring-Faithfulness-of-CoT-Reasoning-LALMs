<<<<<<< HEAD
#!/usr/bin/env python3
# test_mistral_inference.py

"""
Standalone test script to verify Mistral Small 3 is working correctly using vLLM.

This script:
1. Loads Mistral Small 3 using vLLM
=======
# test_mistral_inference.py

"""
Standalone test script to verify Mistral Small 3 is working correctly.

This script:
1. Loads Mistral Small 3 using native HuggingFace Transformers
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
2. Runs a sample mistake generation
3. Runs a sample paraphrasing
4. Prints outputs and timing information

Usage:
    python test_mistral_inference.py [--model-path /path/to/local/model]
"""

import argparse
import time
import logging
import sys

# Add project root to path
<<<<<<< HEAD
sys.path.insert(0, '/project/def-csubakan-ab/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs')
=======
sys.path.insert(0, '/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs')
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec

from core.mistral_utils import (
    load_mistral_model,
    run_mistral_inference,
    generate_mistake,
    paraphrase_text
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


<<<<<<< HEAD
def test_basic_inference(model):
=======
def test_basic_inference(model, tokenizer):
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    """Test basic chat inference."""
    print("\n" + "="*60)
    print("TEST 1: Basic Inference")
    print("="*60)
    
    messages = [
        {"role": "user", "content": "What is 2 + 2? Answer briefly."}
    ]
    
    start_time = time.time()
<<<<<<< HEAD
    response = run_mistral_inference(model, messages, max_new_tokens=50)
=======
    response = run_mistral_inference(model, tokenizer, messages, max_new_tokens=50)
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    elapsed = time.time() - start_time
    
    print(f"Prompt: {messages[0]['content']}")
    print(f"Response: {response}")
    print(f"Time: {elapsed:.2f}s")
    
    return response is not None and len(response) > 0


<<<<<<< HEAD
def test_mistake_generation(model):
=======
def test_mistake_generation(model, tokenizer):
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    """Test mistake generation for CoT reasoning."""
    print("\n" + "="*60)
    print("TEST 2: Mistake Generation")
    print("="*60)
    
    question = "A train travels at 60 km/h for 2 hours. How far does it travel?"
    choices = "(A): 100 km\n(B): 120 km\n(C): 130 km\n(D): 140 km"
    original_sentence = "The train travels at 60 km/h, so in 2 hours it covers 60 × 2 = 120 km."
    
    print(f"Question: {question}")
    print(f"Original sentence: {original_sentence}")
    
    start_time = time.time()
<<<<<<< HEAD
    mistaken = generate_mistake(model, question, choices, original_sentence)
=======
    mistaken = generate_mistake(model, tokenizer, question, choices, original_sentence)
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    elapsed = time.time() - start_time
    
    print(f"Mistaken sentence: {mistaken}")
    print(f"Time: {elapsed:.2f}s")
    
    return mistaken is not None and len(mistaken) > 0


<<<<<<< HEAD
def test_paraphrasing(model):
=======
def test_paraphrasing(model, tokenizer):
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    """Test paraphrasing capability."""
    print("\n" + "="*60)
    print("TEST 3: Paraphrasing")
    print("="*60)
    
    text = "The audio contains a woman speaking in English about the weather forecast. She mentions that it will be sunny tomorrow with temperatures reaching 25 degrees Celsius."
    
    print(f"Original text: {text}")
    
    start_time = time.time()
<<<<<<< HEAD
    paraphrased = paraphrase_text(model, text)
=======
    paraphrased = paraphrase_text(model, tokenizer, text)
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    elapsed = time.time() - start_time
    
    print(f"Paraphrased text: {paraphrased}")
    print(f"Time: {elapsed:.2f}s")
    
    return paraphrased is not None and len(paraphrased) > 0


def main():
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Test Mistral Small 3 inference with vLLM")
=======
    parser = argparse.ArgumentParser(description="Test Mistral Small 3 inference")
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
<<<<<<< HEAD
        help="Path to local model weights. If not provided, downloads from HuggingFace."
    )
    args = parser.parse_args()
    
    print("="*60)
    print("MISTRAL SMALL 3 INFERENCE TEST (vLLM)")
    print("="*60)
    
    # Load model
    print("\nLoading Mistral Small 3 with vLLM...")
    start_time = time.time()
    model = load_mistral_model(model_path=args.model_path)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Run tests
    results = {}
    results['basic'] = test_basic_inference(model)
    results['mistake'] = test_mistake_generation(model)
    results['paraphrase'] = test_paraphrasing(model)
=======
        help="Path to local model weights. If not provided, uses HuggingFace model ID."
    )
    args = parser.parse_args()
    
    print("="*60, flush=True)
    print("MISTRAL SMALL 3 INFERENCE TEST (Native Transformers)", flush=True)
    print("="*60, flush=True)
    
    # Load model
    print("\nLoading Mistral Small 3 with HuggingFace Transformers...", flush=True)
    start_time = time.time()
    model, tokenizer = load_mistral_model(model_path=args.model_path)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s", flush=True)
    
    # Run tests
    results = {}
    results['basic'] = test_basic_inference(model, tokenizer)
    results['mistake'] = test_mistake_generation(model, tokenizer)
    results['paraphrase'] = test_paraphrasing(model, tokenizer)
>>>>>>> 7917316929f71a0ea4e2725be52fee4dfb0006ec
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed! ✓' if all_passed else 'Some tests failed ✗'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
