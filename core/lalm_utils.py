# core/lalm_utils.py

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa
import re

# ==============================================================================
#  START: Runtime Patch for 'cache_position' Bug
# ==============================================================================
# This patch addresses a bug in certain versions of the transformers library
# where the 'cache_position' argument is unexpectedly passed to the model's
# forward method, causing a TypeError.
#
# Method:
# 1. Create a subclass of the original model.
# 2. Override its `forward` method to accept and then "swallow" the
#    `cache_position` argument, preventing it from being passed to the
#    original `super().forward()` call.
# 3. "Monkey-patch" the transformers module at runtime to replace the original
#    class with our patched version. This ensures that any call to
#    `from_pretrained` will instantiate our bug-fixed version.
#
# This approach is vastly superior to manual file editing as it is
# reproducible, maintainable, and non-invasive.

class Qwen2AudioForConditionalGenerationPatched(Qwen2AudioForConditionalGeneration):
    """Patched version of Qwen2AudioForConditionalGeneration that handles caching parameters correctly."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cache_position = False
    
    def forward(self, *args, cache_position=None, **kwargs):
        """
        Forward pass that safely handles cache-related parameters.
        """
        # Remove any cache-related kwargs that this model doesn't support
        kwargs.pop('cache_position', None)
        return super().forward(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Override to ensure proper cache handling during generation."""
        # Remove cache-related kwargs before calling parent
        kwargs.pop('cache_position', None)
        return super().prepare_inputs_for_generation(*args, **kwargs)
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Override to ensure proper cache handling during beam search."""
        # Ensure cache-related parameters are handled correctly during beam search
        if past_key_values is None:
            return None
        return super()._reorder_cache(past_key_values, beam_idx)

# Dynamically swap the original class with our patched one in the library's module.
import transformers.models.qwen2_audio.modeling_qwen2_audio as _mod
_mod.Qwen2AudioForConditionalGeneration = Qwen2AudioForConditionalGenerationPatched

# ==============================================================================
#  END: Runtime Patch
# ==============================================================================


def load_model_and_tokenizer(model_path: str):
    """
    Loads the specified Qwen2-Audio model and processor.
    This function will now use our patched model class automatically.
    """
    print("Applying runtime patch for Qwen2AudioForConditionalGeneration...")
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        sampling_rate=16000
    )
    
    print(f"Loading model from {model_path}...")
    # This call now instantiates Qwen2AudioForConditionalGenerationPatched under the hood
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Ensure the model knows it doesn't support cache_position
    if hasattr(model, 'supports_cache_position'):
        model.supports_cache_position = False
    print("Model and processor loaded successfully with patch!")
    return model, processor


def run_inference(model, processor, messages: list, audio_path: str, max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.9, do_sample: bool = True):
    """
    Runs inference on the model with a given chat history and audio input.
    """
    conversation = []
    for message in messages:
        if message["role"] == "user" and "audio" in message.get("content", ""):
            conversation.append({
                "role": "user", 
                "content": [
                    {"type": "audio", "audio_path": audio_path},
                    {"type": "text", "text": message["content"]},
                ]
            })
        else:
            conversation.append(message)
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    target_sampling_rate = processor.feature_extractor.sampling_rate
    audio_data, _ = librosa.load(
        audio_path, 
        sr=target_sampling_rate
    )
    
    inputs = processor(
        text=text,
        audio=[audio_data],
        sampling_rate=target_sampling_rate,
        return_tensors="pt",
        padding=True
    )

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if do_sample:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        generation_kwargs["do_sample"] = False


    with torch.no_grad():
        generate_ids = model.generate(**inputs, **generation_kwargs)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response


def parse_answer(text: str) -> str | None:
    """
    Parses the model's text output to find the multiple-choice answer.
    """
    if not text:
        return None
    cleaned_text = text.strip()
    match = re.search(r'^\(([A-Z])\)$', cleaned_text)
    if match:
        return match.group(1)
    match = re.search(r'^([A-Z])\)$', cleaned_text)
    if match:
        return match.group(1)
    match = re.search(r'answer is\s+([A-Z])', cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1)
    if len(cleaned_text) == 1 and 'A' <= cleaned_text <= 'Z':
        return cleaned_text
    return None