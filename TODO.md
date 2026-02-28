# TODO — Prioritized Task List

## 1. Model-Specific Prompt Strategy Refactor ✅

- [x] Add no-reasoning prompt builders to `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [x] Add centralized builder functions to `prompt_strategies.py` (`run_conditioned_trial`, `run_no_reasoning_trial`)
- [x] Fix `early_answering.py` 0% case to use no-reasoning prompt
- [x] Migrate `filler_text_utils.py`, `paraphrasing.py`, `adding_mistakes.py` to use centralized delegation
- [x] Migrate `no_reasoning.py` to use centralized no-reasoning delegation
- [x] Smoke test baseline + early answering for `flamingo_hf` (5 samples, 3 chains) ✓
- [ ] **MANDATORY**: Make `run_conditioned_inference` and `run_no_reasoning_inference` required in ALL model_utils (remove fallback)

## 2. Full-Scale Baseline + Early Answering Runs (CURRENT PRIORITY)

- [ ] Submit baseline `flamingo_hf` — all 5 datasets (mmar, sakura-animal/emotion/gender/language)
- [ ] Submit baseline `qwen_omni` — all 5 datasets
- [ ] Submit early answering `flamingo_hf` — all 5 datasets (after baseline completes)
- [ ] Submit early answering `qwen_omni` — all 5 datasets (after baseline completes)
- [ ] Validate output schema + prompt consistency across all runs

## 3. Filler Text CoT Intervention

- [x] Migrate `filler_text_utils.py` `run_filler_trial()` to use `run_conditioned_trial`
- [ ] Run `no_reasoning` foundational experiment for `flamingo_hf` and `qwen_omni` (dependency for filler 0% case)
- [ ] Smoke test filler text for both models (5 samples)
- [ ] Full-scale filler text runs

## 4. Paraphrasing Experiment

- [x] Migrate `paraphrasing.py` `run_paraphrasing_trial()` to use `run_conditioned_trial`
- [ ] Smoke test paraphrasing for both models (5 samples, self-perturbation)
- [ ] Full-scale paraphrasing runs
- [ ] Clean up external perturbation naming convention:
  - Output: `results/{model}/paraphrasing/paraphrasing_{model}_{dataset}_external_perturbations-mistral.jsonl`
  - Each JSONL row should include `perturbation_source` field (e.g., "self", "mistral")

## 5. Adding Mistakes Experiment

- [x] Migrate `adding_mistakes.py` `run_final_trial()` to use `run_conditioned_trial`
- [x] Add `continue_reasoning` hook for model-specific implementations
- [ ] Implement `run_continue_reasoning` in `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [ ] Smoke test + full-scale runs
- [ ] Clean up external perturbation naming convention (same as paraphrasing)

## 6. Audio Masking Experiments (LATER)

- [ ] Run audio masking for `qwen_omni` and `flamingo_hf`
- [ ] Verify parallel pipeline for new models
