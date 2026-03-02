# TODO — Prioritized Task List

## 1. Model-Specific Prompt Strategy Refactor ✅

- [x] Centralized `run_conditioned_trial`, `run_no_reasoning_trial` in `prompt_strategies.py`
- [x] `run_no_reasoning_inference` in `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [x] `run_continue_reasoning` in `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [x] Migrated: `early_answering.py`, `filler_text.py`, `filler_text_utils.py`, `paraphrasing.py`, `adding_mistakes.py`, `no_reasoning.py`
- [x] Removed forced `do_sample=False` — all generation uses model defaults
- [x] Smoke test baseline + early answering for `flamingo_hf` ✓
- [ ] **MANDATORY**: Make `run_conditioned_inference` and `run_no_reasoning_inference` required in ALL model_utils (remove fallback)

## 2. AF3 (flamingo_hf) Experiments

### Baselines ✅

- [x] Submit baseline `flamingo_hf` — all 5 datasets (jobs 25177439-44, COMPLETED)
- [x] Fixed `sanitized_cot` double-strip bug + post-processed all 5 baseline files

### Downstream Experiments (IN PROGRESS)

- [x] Submit early answering — 5 datasets (jobs 25195099-103)
- [x] Submit random partial filler text — 5 datasets (jobs 25195114-118)
- [ ] Submit Mistral perturbation generation — 10 scripts ready, pending submission
- [ ] Submit `adding_mistakes --use-external-perturbations --external-llm mistral` — after perturbations complete
- [ ] Submit `paraphrasing --use-external-perturbations --external-llm mistral` — after perturbations complete

## 3. Qwen Omni Experiments

### Baselines

- [ ] Submit baseline `qwen_omni` — all 5 datasets with sample range splitting
  - Range splits: 0-150, 150-300, 300-450, ..., etc.
  - Need merge step after all range jobs complete per dataset

### Downstream Experiments

- [ ] Early answering
- [ ] Random partial filler text
- [ ] Mistral perturbation generation
- [ ] Adding mistakes + paraphrasing with external perturbations

## 4. External LLM Perturbation Pipeline ✅

- [x] Refactored directory structure: `results/external_llm_perturbations/{llm}/{model}/{dataset}/raw/`
- [x] Dynamic `--external-llm` flag (supports `mistral`, `llama`)
- [x] Auto-constructed paths from CLI args (no mandatory JSONL path)
- [x] Mistral uses vLLM defaults (temperature=1.0, top_p=1.0)
- [x] Added `torch.manual_seed(42)` to `generate_perturbations.py`
- [x] Updated prompts for mistake generation and paraphrasing
- [x] Increased `max_new_tokens` for mistakes: 75 → 256

## 5. Future: LLaMA as External LLM

- [ ] Add LLaMA support to `generate_perturbations.py` (new model loader + inference fn)
- [ ] Create `core/llama_utils.py` for LLaMA-based perturbation generation
- [ ] Generate and submit LLaMA perturbation jobs
- [ ] Compare Mistral vs LLaMA perturbation quality

## 6. Audio Masking Experiments (LATER)

- [ ] Run audio masking for `qwen_omni` and `flamingo_hf`
- [ ] Verify parallel pipeline for new models

## 7. Paraphrasing Experiment

- [x] Migrated `run_paraphrasing_trial()` to `run_conditioned_trial`
- [x] Added 0% case (run conditioned inference with original CoT, not skipped)
- [x] Added `perturbation_source` field to output JSONL rows (`self` / `external-mistral`)
- [ ] Smoke test paraphrasing for both models (5 samples, self-perturbation)
- [ ] Full-scale paraphrasing runs

## 8. Adding Mistakes Experiment

- [x] Migrated `run_final_trial()` to `run_conditioned_trial`
- [x] `continue_reasoning` → dispatches to `run_continue_reasoning` if available
- [x] Implemented `run_continue_reasoning` in `qwen_omni_utils.py` and `audio_flamingo_hf_utils.py`
- [ ] Add `perturbation_source` field to output JSONL rows
- [ ] Smoke test + full-scale runs
