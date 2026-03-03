# Experiments Guidelines & Status Tracker

## Models

- **Audio Flamingo 3** (`flamingo_hf`) → env: `af3_new_hf_env`, modules: `StdEnv/2023 cuda rust gcc arrow`
- **Qwen Omni** (`qwen_omni`) → env: `qwen_omni_env`, modules: `StdEnv/2023 cuda rust gcc arrow`
- **Mistral** (perturbation generation only) → env: `mistral_env`, modules: `StdEnv/2023 cuda rust gcc arrow opencv`

## Datasets

`mmar`, `sakura-animal`, `sakura-emotion`, `sakura-gender`, `sakura-language`
(MMAU to be added later)

## GPU Requirements

| Task                             | GPU                                          |
| -------------------------------- | -------------------------------------------- |
| All model experiments (AF3/Qwen) | `nvidia_h100_80gb_hbm3_3g.40gb:1` (40GB MIG) |
| Mistral perturbation generation  | `h100:1` (full 80GB)                         |

## Global Flags (all experiments)

- `--verbose`
- `--num-chains 1`

---

## Phase 0: Baseline (MMAR only — both models)

| Model         | Dataset | Account        | Splits                                    | Time per part |
| ------------- | ------- | -------------- | ----------------------------------------- | ------------- |
| `flamingo_hf` | `mmar`  | `rrg-csubakan` | 5 parts (`--start-sample`/`--end-sample`) | `01:15:00`    |
| `qwen_omni`   | `mmar`  | `rrg-csubakan` | 5 parts (`--start-sample`/`--end-sample`) | `01:15:00`    |

**Note:** Baseline also needed for MMAU later (same account, same split strategy).

**CLI:** `python main.py --model {MODEL} --experiment baseline --dataset mmar --start-sample S --end-sample E --num-chains 1 --verbose`

---

## Phase 1: Mistral Perturbation Generation (depends on Phase 0)

**Dependency:** Requires baseline results to exist first.

| Model | Dataset | Mode         | Account        | Time       |
| ----- | ------- | ------------ | -------------- | ---------- |
| both  | all 5   | `mistakes`   | `rrg-ravanelm` | `01:00:00` |
| both  | all 5   | `paraphrase` | `rrg-ravanelm` | `01:40:00` |

**GPU:** `h100:1` (full 80GB)

**CLI:** `python scripts/generate_perturbations.py --model {MODEL} --dataset {DATASET} --mode {mistakes|paraphrase}`

---

## Phase 2: CoT Intervention Experiments

_Datasets: all 5 — mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language_
_All per-dataset, per-model scripts_

### Early Answering (no splits needed — single job per dataset)

| Model         | Account           | Time per dataset |
| ------------- | ----------------- | ---------------- |
| `flamingo_hf` | `def-csubakan-ab` | `01:20:00`       |
| `qwen_omni`   | `def-csubakan-ab` | `01:20:00`       |

**CLI:** `python main.py --model {MODEL} --experiment early_answering --dataset {DATASET} --num-chains 1 --verbose`

### Adding Mistakes — 5 parts per dataset (`--part`/`--total-parts`)

| Model         | Account        | Time per part |
| ------------- | -------------- | ------------- |
| `flamingo_hf` | `rrg-ravanelm` | `00:55:00`    |
| `qwen_omni`   | `rrg-csubakan` | `01:00:00`    |

**Requires:** Splitting baseline first + Mistral `mistakes` perturbations done

**CLI:** `python main.py --model {MODEL} --experiment adding_mistakes --dataset {DATASET} --use-external-perturbations --external-llm mistral --part P --total-parts 5 --num-chains 1 --verbose`

### Random Partial Filler Text (lorem) — 5 parts per dataset (`--part`/`--total-parts`)

| Model         | Account        | Time per part |
| ------------- | -------------- | ------------- |
| `flamingo_hf` | `rrg-csubakan` | `01:00:00`    |
| `qwen_omni`   | `rrg-csubakan` | `01:00:00`    |

**CLI:** `python main.py --model {MODEL} --experiment random_partial_filler_text --dataset {DATASET} --filler-type lorem --part P --total-parts 5 --num-chains 1 --verbose`

### Paraphrasing (no splits needed — single job per dataset)

| Model         | Account        | Time per dataset |
| ------------- | -------------- | ---------------- |
| `flamingo_hf` | `rrg-csubakan` | `01:30:00`       |
| `qwen_omni`   | `rrg-csubakan` | `02:00:00`       |

**Requires:** Mistral `paraphrase` perturbations done

**CLI:** `python main.py --model {MODEL} --experiment paraphrasing --dataset {DATASET} --use-external-perturbations --external-llm mistral --num-chains 1 --verbose`

---

## Phase 3: Future — MMAU

- Baseline + all intervention experiments for both models on MMAU dataset.

---

## SLURM Account Summary

| Experiment            | AF3 Account       | Qwen Account      |
| --------------------- | ----------------- | ----------------- |
| Baseline (mmar)       | `rrg-csubakan`    | `rrg-csubakan`    |
| Mistral Generation    | `rrg-ravanelm`    | `rrg-ravanelm`    |
| Early Answering       | `def-csubakan-ab` | `def-csubakan-ab` |
| Adding Mistakes       | `rrg-ravanelm`    | `rrg-csubakan`    |
| Random Partial Filler | `rrg-csubakan`    | `rrg-csubakan`    |
| Paraphrasing          | `rrg-csubakan`    | `rrg-csubakan`    |

---

## Script Requirements

### Resource Monitoring (background, every 20 min)

Must report: RAM (total/used/available), CPU (cores allocated, usage %), GPU (utilization %, VRAM used/total, temperature °C), plus any other useful metrics.

### Runtime Summary (end of script)

Must report: **Start Time** (with timezone), **End Time** (with timezone), **Duration** in `HH:MM:SS` format.

---

## Dependency Chain

```
Phase 0: Baseline (mmar)
    ├──→ Phase 1: Mistral Perturbations (all 5 datasets × both models × 2 modes)
    │        ├──→ Phase 2: Adding Mistakes
    │        └──→ Phase 2: Paraphrasing
    ├──→ Phase 2: Early Answering (depends only on baseline)
    └──→ Phase 2: Random Partial Filler Text (depends only on baseline)
```

---

## Parallelization Pipeline (Split → Run → Verify → Merge)

### Step 1 — Split baseline for parallel runs

```bash
python data_processing/split_dataset_for_parallel_runs.py \
  --model {MODEL} --dataset {DATASET} --num-parts 5 --skip-no-reasoning
```

Creates: `baseline_{MODEL}_{DATASET}.part_{1..5}.jsonl` in `results/{MODEL}/baseline/`

### Step 2 — Run parallel jobs

Scripts use `--part ${SLURM_ARRAY_TASK_ID} --total-parts 5` (handled by SLURM array `1-5`).

### Step 3 — Verify completeness

```bash
# For random partial filler (lorem):
python data_processing/verify_parallel_completeness.py \
  --model {MODEL} --experiment random_partial_filler_text --dataset {DATASET} \
  --expected-parts 5 --num-chains 1 --filler-type lorem

# For adding mistakes (mistral):
python data_processing/verify_parallel_completeness.py \
  --model {MODEL} --experiment adding_mistakes --dataset {DATASET} \
  --expected-parts 5 --num-chains 1 --perturbation-source mistral
```

### Step 4 — Merge parts into final file

```bash
# For random partial filler (lorem):
python data_processing/merge_parallel_results.py \
  --model {MODEL} --experiment random_partial_filler_text --dataset {DATASET} \
  --expected-parts 5 --num-chains 1 --filler-type lorem

# For adding mistakes (mistral):
python data_processing/merge_parallel_results.py \
  --model {MODEL} --experiment adding_mistakes --dataset {DATASET} \
  --expected-parts 5 --num-chains 1 --perturbation-source mistral
```

---

## Pre-Submission TODO

- [ ] Split baseline results for parallel experiments
- [ ] Create log directories: `mkdir -p logs/{model}/{experiment}`
