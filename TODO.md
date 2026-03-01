# TODO — Prioritized Task List

## 1. Model-Specific Prompt Strategy Refactor ✅

- [x] Centralized `run_conditioned_trial`, `run_no_reasoning_trial` in `prompt_strategies.py`
- [x] `run_no_reasoning_inference` in `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [x] `run_continue_reasoning` in `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [x] Migrated: `early_answering.py`, `filler_text.py`, `filler_text_utils.py`, `paraphrasing.py`, `adding_mistakes.py`, `no_reasoning.py`
- [x] Removed forced `do_sample=False` — all generation uses model defaults
- [x] Smoke test baseline + early answering for `flamingo_hf` ✓
- [ ] **MANDATORY**: Make `run_conditioned_inference` and `run_no_reasoning_inference` required in ALL model_utils (remove fallback)

## 2. Full-Scale Baseline + Parallelized Experiments (CURRENT PRIORITY)

## 2. Full-Scale Baseline + Parallelized Experiments (CURRENT PRIORITY)

## 2. Full-Scale Baseline + Early Answering & Filler Text (CURRENT PRIORITY)

- [x] Submit baseline `flamingo_hf` — all 5 datasets (mmar, sakura-animal/emotion/gender/language) -> **(Submitted jobs 25177439-44)**
- [x] Submit early answering `flamingo_hf` — all 5 datasets (1 standalone script per dataset, 3.5h, `rrg-ravanelm`)
- [x] Submit random partial filler text `flamingo_hf` — all 5 datasets (1 standalone script per dataset, 3.5h, `rrg-csubakan`)
- [ ] Submit baseline `qwen_omni` — all 5 datasets (pending demo completion)

## 3. Filler Text CoT Intervention

- [x] Migrated `run_filler_trial()` to `run_conditioned_trial`
- [x] Removed `no_reasoning` dependency — 0% runs actual conditioned inference with full original CoT
- [x] Fixed `create_word_level_masked_cot` input (was passing spaces, now passes actual CoT text)

## 4. Paraphrasing Experiment

- [x] Migrated `run_paraphrasing_trial()` to `run_conditioned_trial`
- [x] Added 0% case (run conditioned inference with original CoT, not skipped)
- [x] Added `perturbation_source` field to output JSONL rows (`self` / `external-mistral`)
- [ ] Smoke test paraphrasing for both models (5 samples, self-perturbation)
- [ ] Full-scale paraphrasing runs

## 5. Adding Mistakes Experiment

- [x] Migrated `run_final_trial()` to `run_conditioned_trial`
- [x] `continue_reasoning` → dispatches to `run_continue_reasoning` if available
- [x] Implemented `run_continue_reasoning` in `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [ ] Add `perturbation_source` field to output JSONL rows
- [ ] Smoke test + full-scale runs

## 6. Audio Masking Experiments (LATER)

- [ ] Run audio masking for `qwen_omni` and `flamingo_hf`
- [ ] Verify parallel pipeline for new models
