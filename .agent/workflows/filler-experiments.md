---
description: Run remaining filler experiments and Mistral perturbations - check status and execute next steps
---

# Filler Experiments Workflow - UPDATED

## Current Status (Dec 29, 2025)

### ✅ ALL QWEN & SALMONN EXPERIMENTS COMPLETE

| Model           | Filler (Dots) | Filler (Lorem) | Mistral Adding Mistakes | Mistral Paraphrasing |
| --------------- | ------------- | -------------- | ----------------------- | -------------------- |
| **Qwen**        | ✅ Done       | ✅ Done        | ✅ Done                 | ✅ Done              |
| **SALMONN 7B**  | ✅ Done       | ✅ Done        | ✅ Done                 | ✅ Done              |
| **SALMONN 13B** | ✅ Done       | ✅ Done        | N/A                     | N/A                  |

All plots generated for the above experiments.

---

## Resource Allocation Reference

### Available Accounts

- `rrg-ravanelm`
- `rrg-csubakan`
- `def-csubakan-ab`

### GPU Options

- **40GB Slice:** `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1`
- **Full H100 (Mistral):** `--gpus=h100:1`

---

# REMAINING: AF3 (Audio Flamingo 3)

## Step 1: Mistral Perturbation Generation

### Allocate Node (Full H100)

```bash
salloc --time=02:00:00 --gpus=h100:1 --cpus-per-task=2 --mem=64G --account=rrg-ravanelm
```

### Load Environment

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow rust opencv
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/mistral_env/bin/activate
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs
```

### Generate Perturbations (5 datasets × 2 modes = 10 files)

```bash
# Mistakes
for dataset in mmar sakura-animal sakura-emotion sakura-gender sakura-language; do
  time python scripts/generate_perturbations.py \
    --baseline-results results/flamingo/baseline/baseline_flamingo_${dataset}-restricted.jsonl \
    --output results/mistral_perturbations/flamingo_${dataset}-restricted_mistakes.jsonl \
    --mode mistakes
done

# Paraphrasing
for dataset in mmar sakura-animal sakura-emotion sakura-gender sakura-language; do
  time python scripts/generate_perturbations.py \
    --baseline-results results/flamingo/baseline/baseline_flamingo_${dataset}-restricted.jsonl \
    --output results/mistral_perturbations/flamingo_${dataset}-restricted_paraphrasing.jsonl \
    --mode paraphrase
done
```

---

## Step 2: Combine Baseline with Perturbations

```bash
module load StdEnv/2023 gcc/12.3 python/3.11
source analysis_env/bin/activate
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

for dataset in mmar sakura-animal sakura-emotion sakura-gender sakura-language; do
  python scripts/combine_baseline_with_perturbations.py --model flamingo --dataset $dataset --experiment adding_mistakes --restricted
  python scripts/combine_baseline_with_perturbations.py --model flamingo --dataset $dataset --experiment paraphrasing --restricted
done
```

---

## Step 3: AF3 Filler Experiments

### Allocate Node

```bash
salloc --time=01:00:00 --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=2 --mem=64G --account=rrg-csubakan
```

### Environment

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.6 arrow python/3.11
source /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/audio-flamingo-env/bin/activate
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs
```

### Demo Commands (time each to estimate full job duration)

```bash
# Partial Dots
time python main.py --model flamingo --experiment partial_filler_text --dataset mmar --restricted --num-samples 5 --verbose

# Flipped Dots
time python main.py --model flamingo --experiment flipped_partial_filler_text --dataset mmar --restricted --num-samples 5 --verbose

# Random Lorem
time python main.py --model flamingo --experiment random_partial_filler_text --dataset mmar --restricted --filler-type lorem --num-samples 5 --verbose

# Partial Lorem
time python main.py --model flamingo --experiment partial_filler_text --dataset mmar --restricted --filler-type lorem --num-samples 5 --verbose

# Flipped Lorem
time python main.py --model flamingo --experiment flipped_partial_filler_text --dataset mmar --restricted --filler-type lorem --num-samples 5 --verbose
```
