# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample


def run_once(model, cfg, wav_processor, wav_path, prompt, device):
    # Prepare samples (verbose True will print more inside prepare_one_sample if you added that)
    samples = prepare_one_sample(wav_path, wav_processor, target_sr=16000, cuda_enabled=False, verbose=True)

    # Print device info for model and tensors
    model_device = next(model.parameters()).device
    print(f"[debug] model device: {model_device}")

    def print_sample_info(samples_dict):
        for k, v in samples_dict.items():
            try:
                dev = v.device
                shape = tuple(v.shape)
            except Exception:
                dev = None
                shape = None
            print(f"[debug] samples['{k}'] type={type(v)} shape={shape} device={dev}")
    print_sample_info(samples)

    # If move_to_cuda doesn't exist or is buggy, ensure tensors are on model device:
    # move spectrogram & raw_wav to model device explicitly
    try:
        samples["spectrogram"] = samples["spectrogram"].to(model_device)
        samples["raw_wav"] = samples["raw_wav"].to(model_device)
        samples["padding_mask"] = samples["padding_mask"].to(model_device)
        print("[debug] Manually moved inputs to model device.")
    except Exception as e:
        print("[debug] Failed to move samples to model device:", e)

    print("Prompt passed to model:", repr(prompt))

    # Show the generation config from cfg (if present)
    gen_cfg = getattr(cfg.config, "generate", None) or getattr(cfg.config, "gen", None) or {}
    print("[debug] cfg.config.generate:", gen_cfg)

    # Safe fallback generation args (override) — use a sensible max_new_tokens to test
    safe_override = {"max_new_tokens": 256, "do_sample": False}
    # Merge safely: values in gen_cfg keep unless conflicting keys exist
    gen_args = dict(gen_cfg) if isinstance(gen_cfg, dict) else {}
    gen_args.update(safe_override)
    print("[debug] generation args used (override):", gen_args)

    # Run generation and print raw result repr
    try:
        # Use explicit autocast for modern torch versions
        with torch.amp.autocast("cuda", dtype=torch.float16) if "cuda" in str(model_device) else torch.no_grad():
            raw_out = model.generate(samples, gen_args, prompts=[prompt])

        print("[debug] raw_out type:", type(raw_out))
        try:
            print("[debug] raw_out repr:", repr(raw_out)[:2000])
        except Exception:
            print("[debug] raw_out cannot be repr'd fully.")
        # If it's a list and contains strings
        if isinstance(raw_out, (list, tuple)):
            print("[debug] raw_out length:", len(raw_out))
            if len(raw_out) > 0:
                print("Output[0]:", raw_out[0])
        else:
            print("Output:", raw_out)

    except Exception as e:
        print("Error during generation:", e)
        import traceback; traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    # New optional args for non-interactive use:
    parser.add_argument("--wav-path", type=str, help="path to wav file to run once")
    parser.add_argument("--prompt", type=str, help="prompt string to use for a single run")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="do not enter interactive loop; exit after single run (requires --wav-path and --prompt)",
    )

    args = parser.parse_args()
    cfg = Config(args)

    # load model
    model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()

    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    # If wav + prompt provided, do a single run and exit (or if --no-interactive used)
    if args.wav_path and args.prompt:
        run_once(model, cfg, wav_processor, args.wav_path, args.prompt, args.device)
        # if the user explicitly asked to exit after single run, do so
        if args.no_interactive:
            return
        # otherwise fallthrough to interactive loop

    if args.no_interactive:
        # If no-interactive was requested but wav/prompt not provided -> error
        print("Error: --no-interactive specified but --wav-path and/or --prompt missing.", file=sys.stderr)
        sys.exit(1)

    # Original interactive loop
    while True:
        try:
            print("=====================================")
            wav_path = input("Your Wav Path:\n").strip()
            prompt = input("Your Prompt:\n").strip()

            if not wav_path:
                print("Empty wav path, try again.")
                continue
            if not prompt:
                print("Empty prompt, try again.")
                continue

            run_once(model, cfg, wav_processor, wav_path, prompt, args.device)

        except KeyboardInterrupt:
            print("\nExiting on user request.")
            break
        except Exception as e:
            # print exception and continue interactive loop
            print("Exception:", e)
            import traceback
            traceback.print_exc()
            # continue to prompt again


if __name__ == "__main__":
    main()
