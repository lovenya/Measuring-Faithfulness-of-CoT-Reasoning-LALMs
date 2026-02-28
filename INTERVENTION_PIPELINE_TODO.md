# Intervention Pipeline TODO (Next Session)

## 1. Push Pending Omni Commit
- Load LFS and push from your shell:
  - `module load git-lfs`
  - `git lfs install`
  - `git push origin main`
- Confirm remote includes commit: `6c237d0`.

## 2. Locked Core Decisions
- Always save both:
  - `generated_cot`
  - `sanitized_cot`
- Always sanitize CoT for intervention input:
  - remove last sentence/line to reduce direct answer leakage.
- Intervention default source:
  - use `sanitized_cot` as `reasoning_for_intervention`.
- Keep one fixed prompt protocol per model alias during intervention runs.
- Keep adding-mistakes/paraphrasing external Mistral perturbation integration for later cleanup.
- For immediate sanity checks:
  - keep old/default perturbation flow for adding-mistakes/paraphrasing.

## 3. Define Unified Intervention Result Schema
- Required output fields:
  - `id`
  - `chain_id`
  - `model`
  - `prompt_protocol`
  - `generated_cot`
  - `sanitized_cot`
  - `baseline_reasoning`
  - `modified_reasoning`
  - `intervention_type`
  - `predicted_choice`
  - `is_correct`
  - `is_consistent_with_baseline`
  - `prediction_changed`

## 4. Add Prompt Protocol Registry
- Add fixed protocol IDs per model, e.g.:
  - `qwen_omni_xml`
  - `qwen_two_turn`
  - `flamingo_hf_single_turn`
- Map model alias -> one protocol for intervention runs.

## 5. Implement Reasoning Extraction Normalizer
- Add helper to create `reasoning_for_intervention` from baseline results:
  - Omni: parse `<Reasoning>...</Reasoning>` and sanitize tail.
  - Others: sanitize generated chain and store as `sanitized_cot`.
- Ensure baseline outputs always contain both:
  - `generated_cot`
  - `sanitized_cot`

## 6. Implement Conditioned Inference Adapter Interface
- Add per-model conditioned function that consumes `modified_reasoning` and returns shared fields.
- First target:
  - `qwen_omni` conditioned path aligned with `pooneh_version/interface/qwen_2.5_omni_conditioned.py`.
- Next target:
  - AF3 conditioned path aligned with `pooneh_version/interface/af3_wrapper_conditioned.py`.

## 7. Add First Intervention Experiment Module
- Create: `experiments/cot/conditioned_intervention.py`.
- Keep restartability key:
  - `(id, chain_id, intervention_type, level)`.
- Scope for first module:
  - baseline reasoning -> optional edit -> conditioned re-answer.
  - no adding-mistakes/paraphrasing external perturbation integration in this step.

## 8. Add Smoke Tests (1 Sample)
- Baseline + conditioned run for `qwen_omni`.
- Baseline + conditioned run for `flamingo_hf`.
- Validate:
  - no crash
  - parser stability
  - schema compatibility with downstream analysis.

## 9. Add Runbook Commands
- Add copy-paste command set for:
  - baseline
  - intervention generation
  - conditioned run
  - quick verification checks.

## 10. Commit Plan
- Commit A:
  - protocol registry + schema + normalizer.
- Commit B:
  - `qwen_omni` conditioned integration.
- Commit C:
  - `flamingo_hf` conditioned integration.
- Commit D:
  - experiment runner + docs/run commands.
