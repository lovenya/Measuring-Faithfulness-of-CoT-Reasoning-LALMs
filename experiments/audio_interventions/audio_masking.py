"""Deprecated alias module for partial_audio_masking.

Kept for backwards compatibility with old direct imports.
"""

import logging

from experiments.audio_interventions.partial_audio_masking import EXPERIMENT_TYPE
from experiments.audio_interventions.partial_audio_masking import run as _run


def run(model, processor, tokenizer, model_utils, data_samples, config):
    logging.warning(
        "DEPRECATED experiment module 'audio_masking' used. "
        "Please migrate to 'partial_audio_masking'."
    )
    return _run(model, processor, tokenizer, model_utils, data_samples, config)
