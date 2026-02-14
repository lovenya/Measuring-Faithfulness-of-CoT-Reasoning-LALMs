---
description: Running analysis and plotting scripts
---

# Running Analysis Scripts

## Environment Setup

Analysis scripts require the `analysis_env` environment:

```bash
# Module load first (if not already done)
module load StdEnv/2023 gcc arrow scipy-stack

# Activate analysis environment
source analysis_env/bin/activate
```

## Directory Structure

```
analysis/
├── per_dataset/           # Single-dataset plots (multiple chains)
├── cross_dataset/         # Cross-dataset aggregated plots
├── misc_scripts/          # Reusable analysis tools
├── extra_scripts/         # One-time explorations
└── utils.py               # Shared utilities
```

## Running Per-Dataset Plots

```bash
cd /scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs

# Audio masking (supports --mask-mode all for comparison plot)
python analysis/per_dataset/plot_audio_masking.py \
    --model qwen --dataset mmar --mask-type silence --mask-mode all

# Other experiments
python analysis/per_dataset/plot_adding_mistakes.py \
    --model qwen --dataset mmar --restricted --show-consistency-curve
```

## Running Cross-Dataset Plots

```bash
# Audio masking (all combinations)
python analysis/cross_dataset/plot_final_audio_masking.py \
    --model qwen --mask-type all --mask-mode all

# Other experiments
python analysis/cross_dataset/plot_final_adding_mistakes.py \
    --model qwen --restricted
```

## Common Arguments

| Argument       | Description                            |
| -------------- | -------------------------------------- |
| `--model`      | qwen, salmonn, flamingo                |
| `--dataset`    | mmar, sakura-animal, etc. or 'all'     |
| `--restricted` | Use restricted dataset (1-6 step CoTs) |
| `--save-pdf`   | Also save PDF version                  |
| `--show-ci`    | Show 95% confidence interval           |

## Audio Masking Specific

| Argument      | Options                 |
| ------------- | ----------------------- |
| `--mask-type` | silence, noise, all     |
| `--mask-mode` | random, start, end, all |

When `--mask-mode all`: plots all 3 modes as separate lines (solid/dashed/dotted)
