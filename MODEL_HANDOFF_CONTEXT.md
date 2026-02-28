# Model Handoff Context (NKM)

Last updated: 2026-02-28
Repo: `/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs`

This document is a full context transfer for a new model/session to continue work without prior chat memory.
It covers architecture, key codepaths, changes already made, pending changes, result schemas, runbooks, and next macro direction.

---

## 1. Executive Summary

The project has moved from a hard-coded two-turn baseline flow toward a **dynamic prompt strategy framework**, while simultaneously integrating **Pooneh-style conditioned inference workflows** for:

1. `qwen_omni` (Qwen2.5 Omni HF backend)
2. `flamingo_hf` (Audio Flamingo 3 HF backend)



Major direction:

1. Keep the existing robust, model-agnostic pipeline (`main.py`, experiment folders, resumability) intact.
2. Integrate model-specific prompting/parsing/conditioning behavior where needed.
3. Stabilize intervention experiments (starting with `early_answering`) so they are schema-consistent and restartable.
4. Preserve old outputs while running clean new validations.

---

## 2. Current Repo State (Critical)

### 2.1 Working tree status

Current `git status --short`:

1. `M .gitignore`
2. `M experiments/cot/early_answering.py`
3. `? audio-flamingo-code` (submodule dirty marker)
4. `?? INTERVENTION_PIPELINE_TODO.md`

Implication: there are **uncommitted local edits**. The next model should inspect before committing.

### 2.2 Results archival change

`results/` was moved to `old_results/` and a fresh `results/` directory was created.

Current `.gitignore` includes both:

1. `results/`
2. `old_results/`

`old_results/` contains historical large outputs and chunk files (many models/datasets/experiments).

### 2.3 Recent commits (context)

Notable in recent log:

1. `5aab0da` — conditioned early-answering flow for Omni and AF3
2. many subsequent merges/uploads by user

Do not assume only one linear branch of changes; inspect files directly.

---

## 3. Top-Level Architecture

Entry point:

1. `main.py`

Core contracts:

1. `core/*_utils.py` per model backend exposes:
   - `load_model_and_tokenizer`
   - `run_inference`
   - `run_text_only_inference`
   - `sanitize_cot`
   - `parse_answer`
   - `format_choices_for_prompt`
   - optional `run_conditioned_inference` for intervention pipelines

Experiment organization:

1. `experiments/baseline/*`
2. `experiments/cot/*`
3. `experiments/audio_interventions/*`
4. `experiments/testing/*`

Experiment types in each module:

1. `EXPERIMENT_TYPE = "foundational" | "dependent" | "independent"`

`main.py` behavior:

1. Parses CLI and sets `config` globals.
2. Resolves output path conventions (including masks, restricted flags, part files).
3. Dynamically imports model utility module by `--model` alias.
4. Dynamically imports experiment module from subfolders.
5. Applies guardrails:
   - foundational experiments cannot use `--restricted`
   - foundational experiments cannot use `--part`
6. Routes execution by experiment type.

---

## 4. Model Aliases and Backends

Defined in `config.py`:

1. `qwen` -> `core/qwen_utils.py`
2. `qwen_omni` -> `core/qwen_omni_utils.py`
3. `flamingo` -> `core/audio_flamingo_utils.py` (old llava/code backend)
4. `flamingo_hf` -> `core/audio_flamingo_hf_utils.py` (new transformers/hf backend)
5. `salmonn`, `salmonn_7b` -> `core/salmonn_utils.py`

Important direction:

1. `flamingo_hf` is the preferred new AF3 path for this integration cycle.
2. Legacy `flamingo` remains for backward compatibility.

---

## 5. Dynamic Prompt Strategy Migration

### 5.1 Central module

`core/prompt_strategies.py`

Current canonical strategies:

1. `two_turn_sanitized_cot`
2. `single_turn_explicit_letter`

Deprecated aliases still accepted:

1. `legacy_two_turn` -> `two_turn_sanitized_cot`
2. `pooneh_single_turn` -> `single_turn_explicit_letter`

### 5.2 How baseline/audio_masking call it

`experiments/baseline/baseline.py` and `experiments/audio_interventions/audio_masking.py` call:

1. `get_prompt_strategy(config)`
2. `run_reasoning_trial(...)`

This lets one CLI flag switch prompt logic while preserving experiment orchestration.

### 5.3 Strategy semantics

`two_turn_sanitized_cot`:

1. generate CoT with sampling
2. sanitize CoT
3. ask final answer deterministically using sanitized CoT

`single_turn_explicit_letter`:

1. one prompt requesting step-by-step reasoning + explicit final letter
2. parse letter from same response
3. stores full response as both `generated_cot` and `final_answer_raw`, and sanitized variant via `sanitize_cot`

### 5.4 CLI exposure

`main.py --prompt-strategy` currently supports:

1. `two_turn_sanitized_cot`
2. `single_turn_explicit_letter`
3. aliases: `legacy_two_turn`, `pooneh_single_turn`

---

## 6. Pooneh Integration: What Was Ported and Why

Reference scripts in `pooneh_version/interface/`:

1. `qwen_2.5_omni_wrapper.py`
2. `qwen_2.5_omni_conditioned.py`
3. `af3_wrapper.py`
4. `af3_wrapper_conditioned.py`

These scripts influenced:

1. conditioned prompt styles
2. parser behavior
3. inference patterns for Omni and AF3

### 6.1 Qwen Omni port details

In `core/qwen_omni_utils.py`:

1. `run_inference` builds chat template with system prompt + user text/audio.
2. `run_conditioned_inference` uses Pooneh-style XML conditioned prompt:
   - instruction (question + choices)
   - XML template with `<Reasoning>...` and `<Conclusion>...`
3. parsing:
   - primary: `<Conclusion>LETTER</Conclusion>`
   - fallback: trailing letter extraction

Dependency note:

1. Uses `process_mm_info` imported from `qwen_omni_utils` package/module available in environment.
2. If missing, raises explicit ImportError instructing to install dependencies.

### 6.2 AF3 HF port details

In `core/audio_flamingo_hf_utils.py`:

1. strict local model loading (`config.MODEL_PATHS["flamingo_hf"]`)
2. requires `think/adapter_model.safetensors` and `think/non_lora_trainables.bin`
3. loads base + non-lora weights + PEFT adapter
4. input dtype/device handling avoids float/half mismatch
5. conditioned inference prompt style aligned with Pooneh AF3 conditioned script:
   - question + choices
   - “Please think and reason ...”
   - injected provided reasoning
   - “Therefore, the answer is:”

Parser is more robust (pattern + semantic fallback) due AF3 output variability.

---

## 7. Early Answering Intervention Pipeline (Current Core)

File: `experiments/cot/early_answering.py`

### 7.1 Logic

Per baseline trial:

1. load `sanitized_cot`
2. sentence tokenize
3. iterate `k = 0..N` where:
   - `k` is `num_sentences_provided`
   - `modified_reasoning = first k sentences`
4. call model-specific `run_conditioned_inference(..., provided_reasoning=modified_reasoning)`
5. compare with baseline prediction and correct answer
6. write one row per `(id, chain_id, k)`

Restartability key:

1. `(id, chain_id, num_sentences_provided)`

### 7.2 Current output schema order (locked in file)

Current `final_ordered_result` keys:

1. `id`
2. `chain_id`
3. `num_sentences_provided`
4. `total_sentences_in_chain`
5. `predicted_choice`
6. `correct_choice`
7. `is_correct`
8. `corresponding_baseline_predicted_choice`
9. `is_consistent_with_baseline`
10. `final_answer_raw`
11. `modified_reasoning`
12. `final_prompt_messages`
13. `sanitized_cot`
14. `audio_path`
15. `question`
16. `choices`
17. `model`
18. `prompt_protocol`

Notably removed relative to earlier drafts/runs:

1. `generated_cot`
2. `intervention_type`

### 7.3 Important caveat about existing `results/.../early_answering`

Current file `results/qwen_omni/early_answering/early_answering_qwen_omni_mmar.jsonl` has mixed prompt shapes because it contains rows produced before and after prompt-path adjustments.

Observed diagnostic:

1. total rows: 24
2. rows with `<Reasoning>` in final prompt text: 18
3. rows without `<Reasoning>`: 6

Action: rerun clean file for trustworthy interpretation.

---

## 8. Baseline and Independent Experiment Schemas

### 8.1 Baseline schema

`experiments/baseline/baseline.py` writes keys:

1. `id`
2. `chain_id`
3. `predicted_choice`
4. `correct_choice`
5. `is_correct`
6. `final_answer_raw`
7. `final_prompt_messages`
8. `question`
9. `choices`
10. `audio_path`
11. `generated_cot`
12. `sanitized_cot`

### 8.2 Audio masking schema

`experiments/audio_interventions/audio_masking.py`:

For `mask_percent=0`:

1. copies baseline answer (hardcoded anchor)
2. `is_consistent_with_baseline=True`

For `mask_percent>0`:

1. runs trial on masked audio
2. includes `generated_cot` and `sanitized_cot`

---

## 9. Parallelization Hardening Work

Completed scripts:

1. `data_processing/verify_parallel_completeness.py`
2. `data_processing/merge_parallel_results.py`

### 9.1 `verify_parallel_completeness.py`

Capabilities:

1. exact part scan for `1..expected_parts`
2. per-part report columns:
   - `part`, `baseline_trials`, `expected_lines`, `actual_lines`, `status`, `reason`
3. strict expected count for audio masking:
   - expected lines = baseline trials (`chain_id < num_chains`) * expected entries/sample
   - default expected entries/sample for audio masking: 11
4. supports `--json`

### 9.2 `merge_parallel_results.py`

By default:

1. validates with verifier first
2. refuses merge on missing/incomplete chunks
3. exit code `1` on refusal

Override:

1. `--force-merge` merges existing parts with warning

Audio masking aware pathing handled in both scripts:

1. directory: `results/{model}/audio_masking/{mask_type}/{mask_mode}`
2. filename suffix: `_{mask_type}_{mask_mode}`

---

## 10. Environment and Compute Node Workflow

No conda assumption in this repo workflow; venv-based activation is used in practice.

### 10.1 Compute node

1. workflow doc: `.agent/workflows/get_compute_node.md`
2. helper script: `scripts/get_compute_node.sh`
3. default allocation:
   - `--time=02:00:00`
   - `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1`
   - `--cpus-per-task=4`
   - `--mem=64G`
   - `--account=rrg-ravanelm`

### 10.2 Environment activation

1. workflow doc: `.agent/workflows/activate_env.md`
2. helper script: `scripts/activate_env.sh`
3. supports env keys:
   - `qwen`
   - `salmonn`
   - `salmonn_7b`
   - `flamingo`
   - `flamingo_hf`
   - `mistral`
   - `analysis`

### 10.3 Relevant venvs in repo root

Observed:

1. `qwen_omni_env`
2. `af3_new_hf_env`
3. `qwen_new_env`
4. `salmonn_env`
5. `analysis_env`
6. `mistral_env`

---

## 11. Known Issues and Lessons Learned

### 11.1 Qwen Omni processor/load issue previously seen

Past crash:

1. `TypeError: argument of type 'NoneType' is not iterable` during processor load in certain env setups

Current mitigation path:

1. use the correct environment (`qwen_omni_env`)
2. ensure model path and transformers build are compatible

### 11.2 AF3 dtype mismatch previously seen

Past crash:

1. `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`

Fix in `core/audio_flamingo_hf_utils.py`:

1. `_move_inputs_to_model_dtype` casts floating tensors to model dtype while preserving integer tensors.

### 11.3 Old result-file mixing

Because rows were appended across code iterations, some files mix old/new schema or prompt protocols.

Policy now:

1. archive history in `old_results/`
2. run fresh in `results/`
3. avoid schema interpretation on mixed legacy files

### 11.4 Pending local edits mismatch with previous TODO

`INTERVENTION_PIPELINE_TODO.md` still includes older decisions (for example keeping `generated_cot` in intervention outputs), while current `early_answering.py` schema removed it.

Treat code as source of truth unless TODO updated.

---

## 12. File-by-File “Must Read” Map for New Model

### 12.1 Orchestration and config

1. `main.py`
2. `config.py`

### 12.2 Prompt strategy and backends

1. `core/prompt_strategies.py`
2. `core/qwen_omni_utils.py`
3. `core/audio_flamingo_hf_utils.py`
4. `core/qwen_utils.py` (Qwen2-Audio)
5. `core/audio_flamingo_utils.py` (legacy Flamingo backend)
6. `core/salmonn_utils.py`

### 12.3 Foundational experiments

1. `experiments/baseline/baseline.py`
2. `experiments/baseline/no_reasoning.py`

### 12.4 Intervention and CoT experiments

1. `experiments/cot/early_answering.py`
2. `experiments/cot/adding_mistakes.py`
3. `experiments/cot/paraphrasing.py`
4. filler variants:
   - `filler_text.py`
   - `partial_filler_text.py`
   - `random_partial_filler_text.py`
   - `flipped_partial_filler_text.py`

### 12.5 Audio interventions

1. `experiments/audio_interventions/audio_masking.py`
2. `experiments/audio_interventions/adversarial.py`
3. `experiments/audio_interventions/snr_robustness.py`
4. `experiments/audio_interventions/jasco_masking.py`

### 12.6 Parallel workflow tools

1. `data_processing/split_dataset_for_parallel_runs.py`
2. `data_processing/verify_parallel_completeness.py`
3. `data_processing/merge_parallel_results.py`

### 12.7 Pooneh reference implementations

1. `pooneh_version/interface/qwen_2.5_omni_wrapper.py`
2. `pooneh_version/interface/qwen_2.5_omni_conditioned.py`
3. `pooneh_version/interface/af3_wrapper.py`
4. `pooneh_version/interface/af3_wrapper_conditioned.py`

---

## 13. Result Files and Logs Layout

### 13.1 Current active results (fresh)

Observed files currently under `results/`:

1. `results/qwen_omni/baseline/baseline_qwen_omni_mmar.jsonl`
2. `results/qwen_omni/early_answering/early_answering_qwen_omni_mmar.jsonl`
3. logs under `results/logs/qwen_omni/...`

### 13.2 Historical results

Large legacy corpus in `old_results/` including:

1. baseline/no_reasoning/cot/audio_interventions for multiple models
2. many parallel part files (`.part_N.jsonl`)
3. combined perturbation artifacts

---

## 14. Recommended Validation Commands (Fresh Runs)

### 14.1 Qwen Omni smoke

```bash
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs
module load StdEnv/2023 cuda rust gcc arrow

deactivate 2>/dev/null || true
source qwen_omni_env/bin/activate
export QWEN_OMNI_LOCAL_MODEL_PATH=/scratch/lovenya/models/Qwen/Qwen2.5-Omni-7B

rm -f results/qwen_omni/baseline/baseline_qwen_omni_mmar.jsonl
rm -f results/qwen_omni/early_answering/early_answering_qwen_omni_mmar.jsonl

python main.py --model qwen_omni --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --prompt-strategy single_turn_explicit_letter --verbose
python main.py --model qwen_omni --dataset mmar --experiment early_answering --num-samples 1 --num-chains 1 --verbose
```

### 14.2 AF3 HF smoke

```bash
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs
module load StdEnv/2023 cuda rust gcc arrow

deactivate 2>/dev/null || true
source af3_new_hf_env/bin/activate

rm -f results/flamingo_hf/baseline/baseline_flamingo_hf_mmar.jsonl
rm -f results/flamingo_hf/early_answering/early_answering_flamingo_hf_mmar.jsonl

python main.py --model flamingo_hf --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --prompt-strategy single_turn_explicit_letter --verbose
python main.py --model flamingo_hf --dataset mmar --experiment early_answering --num-samples 1 --num-chains 1 --verbose
```

### 14.3 Quick conformance audit

```bash
python - <<'PY'
import json,re
from pathlib import Path

for model in ['qwen_omni','flamingo_hf']:
    p = Path(f'results/{model}/early_answering/early_answering_{model}_mmar.jsonl')
    if not p.exists():
        print(model, 'missing')
        continue
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    print('\n', model, 'rows', len(rows))
    ks={(r['id'],r['chain_id']):set() for r in rows}
    for r in rows:
        ks[(r['id'],r['chain_id'])].add(r['num_sentences_provided'])
    print('chains', len(ks))

    mismatch=0
    for r in rows:
        raw=r.get('final_answer_raw','')
        pred=r.get('predicted_choice')
        if model=='qwen_omni':
            m=re.search(r'<Conclu.*?>\s*([A-Za-z])\s*</Conclu.*?>', raw, re.I)
            got=(m.group(1).upper() if m else None)
        else:
            m=re.search(r'\(([A-J])\)', raw[-120:], re.I)
            got=(m.group(1).upper() if m else None)
        if got and pred!=got:
            mismatch+=1
    print('parser mismatches (strict subset check):', mismatch)
PY
```

---

## 15. Macro Direction (Next)

Primary macro direction is **intervention pipeline stabilization and scale-out**, not starting from scratch.

### 15.1 Immediate next macro block

1. Finalize early-answering prompt consistency and verify fresh clean outputs for both `qwen_omni` and `flamingo_hf`.
2. Freeze early-answering schema and avoid further ad-hoc field churn.
3. Add a shared verifier utility for intervention output consistency.

### 15.2 Next expansion block

After early-answering is stable:

1. Apply same conditioned-inference protocol discipline to:
   - filler text variants
   - audio masking follow-up checks
   - paraphrasing flow
2. Keep one fixed prompt protocol per model alias during each experiment family to avoid confounds.

### 15.3 Parallel demo block

1. Use strict split/verify/merge flow for 5-chunk demos.
2. Validate merge refusal on intentional incomplete chunk.
3. Force merge only as sanity tool, not primary result path.

### 15.4 Analysis block

1. Run pending variance analyses (`experiments/testing/analyze_variance.py`).
2. Continue SNR robustness pipeline after intervention stabilization.

---

## 16. Explicit Open Questions the Next Model Should Resolve Early

1. Should `single_turn_explicit_letter` wording remain exactly current across all models, or split model-specific single-turn strategies?
2. For AF3 conditioned outputs, is current robust parser acceptable or should it be constrained to stricter short-answer extraction for consistency metrics?
3. Should `INTERVENTION_PIPELINE_TODO.md` be rewritten to match current schema decisions (it is currently outdated)?
4. Is `results/qwen_omni/early_answering/...` to be regenerated now that prompt-path logic changed?

---

## 17. Safety / Operational Notes

1. The repo uses large files and LFS; for push operations load `git-lfs` and run `git lfs install`.
2. Existing submodules include `audio-flamingo-code`; working tree currently marks it dirty.
3. Prefer non-destructive operations on old result archives.
4. For compute-node runs, activate the correct venv before invoking `main.py`.

---

## 18. TL;DR for New Session Bootstrapping

If a new model has only 1 minute:

1. Read `main.py`, `config.py`, `core/prompt_strategies.py`, `experiments/cot/early_answering.py`, `core/qwen_omni_utils.py`, `core/audio_flamingo_hf_utils.py`.
2. Note current dirty files from `git status`.
3. Run clean 1-sample baseline + early-answering for `qwen_omni` and `flamingo_hf`.
4. Validate schema/prompt consistency in generated JSONL.
5. Proceed to filler/paraphrasing/audio-masking intervention stabilization.

