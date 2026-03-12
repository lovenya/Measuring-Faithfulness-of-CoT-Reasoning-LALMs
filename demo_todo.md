# Demo TODO (Compute Node Runbook)

This runbook is for validating the Wave 1 refactor on a compute node.
Run from repo root:

```bash
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs
```

## 0) Environment Matrix

Use separate envs as agreed:

- `qwen_audio_2` env
- `qwen_omni_2.5` env
- `salmonn_audio` env (shared for 7B/13B)
- `aflamingo_3` env
- `mistral` env
- `analysis` env

Template:

```bash
source /path/to/<env_name>/bin/activate
python --version
which python
```

## 1) Quick Alias + Dispatch Smoke Test

### 1.1 Canonical aliases

```bash
# Qwen Audio 2
python main.py --model qwen_audio_2 --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --verbose

# Qwen Omni 2.5
python main.py --model qwen_omni_2.5 --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --verbose

# Audio Flamingo 3 (HF backend)
python main.py --model aflamingo_3 --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --verbose

# SALMONN 7B
python main.py --model salmonn_audio_7b --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --verbose

# SALMONN 13B
python main.py --model salmonn_audio_13b --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --verbose
```

### 1.2 Legacy alias warnings (optional check)

```bash
python main.py --model qwen --dataset mmar --experiment baseline --num-samples 1 --num-chains 1 --verbose
python main.py --model qwen_audio_2 --dataset mmar --experiment audio_masking --mask-type silence --mask-mode random --num-samples 1 --num-chains 1 --verbose
```

You should see deprecation warnings for old alias names.

## 2) Prepare Data for Independent Experiments

Independent experiments now run from dataset rows directly and do true inference for all levels.

### 2.1 Generate masked audio data (MMAR demo)

```bash
# Use qwen_audio_2 env (or any env with numpy + soundfile)

for mask_type in silence noise; do
  for mode in start end scattered; do
    python data_processing/mask_audio_dataset.py \
      --source data/mmar \
      --output data/mmar_masked \
      --mask-type "$mask_type" \
      --mode "$mode" \
      --levels 10 20 30 40 50 60 70 80 90 100 \
      --workers 8
  done
done
```

### 2.2 Generate noisy audio data (MMAR demo)

```bash
python data_processing/generate_noisy_audio.py \
  --source data/mmar \
  --output data/mmar_noisy \
  --snr-levels 20 10 5 0 -5 -10 \
  --num-workers 8
```

## 3) Run Demo Experiments (Canonical Names)

Set a consistent demo config:

```bash
MODEL=qwen_audio_2
DATASET=mmar
N_SAMPLES=5
N_CHAINS=2
```

### 3.1 Baseline (needed for analysis-time consistency joins)

```bash
python main.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --experiment baseline \
  --num-samples "$N_SAMPLES" \
  --num-chains "$N_CHAINS" \
  --verbose
```

### 3.2 Partial audio masking (canonical experiment)

```bash
for mask_type in silence noise; do
  for mask_mode in start end scattered; do
    python main.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --experiment partial_audio_masking \
      --mask-type "$mask_type" \
      --mask-mode "$mask_mode" \
      --num-samples "$N_SAMPLES" \
      --num-chains "$N_CHAINS" \
      --verbose
  done
done
```

Expected output tree:

```text
results/{model}/partial_audio_masking/{mask_type}/{mask_mode}/
partial_audio_masking_{model}_{dataset}_{mask_type}_{mask_mode}.jsonl
```

Where:
- `mask_type`: `silence | noise`
- `mask_mode`: `start | end | scattered`

### 3.3 SNR robustness

```bash
python main.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --experiment snr_robustness \
  --num-samples "$N_SAMPLES" \
  --num-chains "$N_CHAINS" \
  --verbose
```

## 4) Run Analysis (analysis env)

Switch to analysis env first (needs at least `pandas`, `matplotlib`, `seaborn`).

### 4.1 SNR summary table

```bash
python analysis/evaluate_snr_robustness.py \
  --model "$MODEL" \
  --datasets "$DATASET" \
  --results-dir results
```

### 4.2 Per-dataset partial audio masking plots

```bash
python analysis/per_dataset/plot_audio_masking.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --mask-type all \
  --mask-mode all \
  --results_dir results \
  --plots_dir plots
```

### 4.3 Cross-dataset partial audio masking plots

```bash
python analysis/cross_dataset/plot_final_audio_masking.py \
  --model "$MODEL" \
  --mask-type all \
  --mask-mode all \
  --results_dir results \
  --plots_dir plots/cross_dataset_plots
```

## 5) Optional: Mistral Perturbation Generation Smoke (mistral env)

```bash
python scripts/generate_perturbations.py --help
```

## 6) Quick Validation Checklist

- Canonical aliases run: `qwen_audio_2`, `qwen_omni_2.5`, `aflamingo_3`, `salmonn_audio_7b`, `salmonn_audio_13b`
- `partial_audio_masking` writes to canonical path tree
- `audio_masking` and `mask-mode random` still run with deprecation warnings
- Independent outputs do not contain precomputed baseline consistency fields
- Analysis computes consistency by baseline join and reports missing joins if any
