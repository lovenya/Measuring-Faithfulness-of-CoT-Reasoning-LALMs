# experiments/flipped_partial_filler_text.py

"""
This script conducts the "Flipped Partial Filler Text" experiment. This is a
variant of the filler text experiments designed to test if the position of
corrupted reasoning matters.

Methodology:
1. It takes a reasoning chain (CoT) from a baseline run.
2. For a given percentage (e.g., 20%), it replaces that percentage of the
   *words* in the CoT with a filler token ("..."). The words to be replaced
   are taken from the END of the reasoning chain.
3. It then presents this partially corrupted CoT to the model and asks for a
   final answer.
4. By comparing the results to the "start" and "random" corruption variants, we
   can determine if the model is more sensitive to errors at the beginning or
   end of its reasoning chain.
"""

import logging
from ._partial_filler_shared import run_partial_filler_experiment

# This is a 'dependent' experiment because it manipulates the CoTs from a 'baseline' run.
EXPERIMENT_TYPE = "dependent"

def run(model, processor, tokenizer, model_utils, config):
    """
    Orchestrates the WORD-LEVEL flipped partial filler text experiment where
    corruption starts from the end of the CoT.
    """
    logging.info("Dispatching WORD-LEVEL Partial Filler (End) experiment to shared runner.")
    run_partial_filler_experiment(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        model_utils=model_utils,
        config=config,
        mode="end",
        position_label="End",
    )