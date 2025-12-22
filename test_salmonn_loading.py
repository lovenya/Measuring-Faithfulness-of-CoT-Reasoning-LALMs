#!/usr/bin/env python3
"""Quick test to verify SALMONN model loading works with the offline fix."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging_utils import setup_logger
import logging
import config

# Initialize logger
log_dir = os.path.join(config.RESULTS_DIR, 'logs')
setup_logger(log_dir, 'salmonn', 'test_loading', 'test')

logging.info("=" * 60)
logging.info("SALMONN Model Loading Test")
logging.info("=" * 60)

# Import SALMONN utils
from core import salmonn_utils

# Get model path
model_path = config.MODEL_PATHS['salmonn_checkpoint']
logging.info(f"Model path: {model_path}")

# Test loading
logging.info("Starting model load test...")
try:
    model, processor, tokenizer = salmonn_utils.load_model_and_tokenizer(model_path)
    logging.info("✓ SUCCESS! Model loaded successfully!")
    logging.info(f"  - Model type: {type(model)}")
    logging.info(f"  - Processor type: {type(processor)}")
    logging.info(f"  - Tokenizer type: {type(tokenizer)}")
    print("\n✓ SALMONN model loading test PASSED!")
except Exception as e:
    logging.exception("✗ FAILED! Model loading failed")
    print(f"\n✗ SALMONN model loading test FAILED: {e}")
    sys.exit(1)
