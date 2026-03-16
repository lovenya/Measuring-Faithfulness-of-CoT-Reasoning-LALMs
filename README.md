# Measuring Faithfulness of CoT Reasoning in Large Audio-Language Models

This repository contains the codebase, experiment drivers, preprocessing utilities, analysis scripts, and companion website assets for our anonymous submission on faithfulness in Large Audio-Language Models (LALMs).

We study whether Chain-of-Thought (CoT) reasoning in LALMs is faithful both to:

- the input audio
- the model's final prediction

To do this, we combine:

- audio interventions, which probe whether models actually listen to the audio
- CoT interventions, which probe whether the generated reasoning is causally connected to the final answer

## Blind Review Note

This repository is currently prepared for blind review.

- Paper link: `TBD`
  - We will add the public paper link after the review process.
- Anonymous companion website: `https://anonymous.4open.science/w/Measuring-Faithfulness-of-CoT-Reasoning-LALMs-A3B6/`
- Anonymous code snapshot: `https://anonymous.4open.science/r/Measuring-Faithfulness-of-CoT-Reasoning-LALMs-A3B6`

The root `index.html` also contains the local website/demo page used to present the benchmark and figures.

## What This Repository Covers

At a high level, this repository supports the full research workflow:

1. prepare environments
2. prepare datasets and model assets
3. preprocess audio / standardized manifests
4. run foundational, audio-intervention, and CoT-intervention experiments
5. analyze results and generate plots
6. publish or inspect the companion website

The codebase is currently being refactored for reproducibility and clearer public release structure, but the main experiment and analysis pipeline is already organized around the workflow above.

## Experiment Overview

### Audio interventions

These experiments test whether the model is faithful to the input audio itself.

- `partial_audio_masking`
  - progressively masks portions of the audio to test holistic listening
- `snr_robustness`
  - adds noise at different SNR levels to test hallucination-free listening
- `adversarial`
  - injects adversarial speech content to test attentive listening
- `jasco_masking`
  - evaluates reliance on spoken text vs environmental audio in a dedicated JASCO pipeline

### CoT interventions

These experiments test whether the generated reasoning is causally important for the final prediction.

- `baseline`
  - generates the reference CoT traces used by most dependent experiments
- `no_reasoning`
  - probes the structural prior of the model without semantic reasoning content
- `early_answering`
  - reveals progressively more of the baseline CoT
- `filler_text`
  - replaces reasoning with filler content to test whether performance comes from meaning or extra computation
- `partial_filler_text`
  - progressively replaces portions of the CoT using `start`, `end`, or `random` corruption modes
- `paraphrasing`
  - rewrites the CoT while preserving meaning
- `adding_mistakes`
  - injects plausible mistakes into the reasoning chain

## Supported Models

Canonical public aliases:

- `qwen_audio_2`
- `qwen_omni_2.5`
- `aflamingo_3`
- `salmonn_audio_7b`
- `salmonn_audio_13b`

Legacy aliases are still accepted internally for back-compat, but the aliases above are the intended public interface.

## Supported Datasets

Primary datasets currently wired through `config.py`:

- `mmar`
- `sakura-animal`
- `sakura-emotion`
- `sakura-gender`
- `sakura-language`
- `jasco`
- `mmau`

Some preprocessing pipelines also generate derived variants such as masked, noisy, restricted, and adversarial-augmented datasets.

## Repository Layout

```text
.
├── main.py                         # Main experiment entrypoint
├── config.py                       # Canonical aliases, paths, and defaults
├── experiments/                    # Foundational, CoT, and audio experiments
├── data_processing/                # Dataset normalization and preprocessing pipelines
├── analysis/                       # Active analysis scripts and archived legacy scripts
├── shell_scripts/                  # Reproducibility helpers and setup scripts
├── data/                           # Standardized datasets and related assets
├── results/                        # Experiment outputs
├── plots/                          # Analysis outputs
├── index.html                      # Companion website entrypoint
└── LALM_Faithfulness/              # Paper source
```

## Reproducibility Workflow

There are two intended ways to use this repository:

- direct CLI usage via `main.py` and the scripts in `data_processing/` and `analysis/`
- helper-driven usage through `shell_scripts/`, which is the intended home for reproducible environment, dataset, and model setup

### 1. Create a Python environment

Environment creation scripts currently available:

- `shell_scripts/envs/create_env_qwen_omni_2.5.sh`
- `shell_scripts/envs/create_env_aflamingo_3.sh`

Example:

```bash
bash shell_scripts/envs/create_env_qwen_omni_2.5.sh
source env/qwen_omni_2.5/bin/activate
```

For more detail, see [shell_scripts/README.md](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/shell_scripts/README.md).

### 2. Prepare model weights and datasets

This repository is designed for local / HPC-first workflows where model weights and datasets live on local storage rather than being fetched at runtime.

Current status:

- environment scripts are already available under `shell_scripts/envs/`
- `shell_scripts/models/` and `shell_scripts/data/` are reserved for model/data setup helpers
- additional download/setup scripts are being finalized as part of the reproducibility refactor

Until those helpers are finalized, dataset and model paths are controlled through [config.py](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/config.py).

### 3. Run preprocessing

Preprocessing scripts live under `data_processing/`. Common tasks include:

- standardizing or enriching datasets
- generating noisy audio
- generating masked audio
- normalizing adversarial and JASCO inputs
- creating restricted subsets
- splitting/merging for parallel runs

Examples:

```bash
python data_processing/mask_audio_dataset.py --help
python data_processing/generate_noisy_audio.py --help
python data_processing/filter_dependent_results_to_restricted_version.py --help
```

### 4. Run experiments

The main entrypoint is:

```bash
python main.py --model <model_alias> --experiment <experiment_name> --dataset <dataset_alias>
```

Examples:

```bash
python main.py --model qwen_omni_2.5 --experiment baseline --dataset mmar
python main.py --model aflamingo_3 --experiment partial_audio_masking --dataset mmar --mask-type silence --mask-mode scattered
python main.py --model qwen_omni_2.5 --experiment snr_robustness --dataset mmar
python main.py --model aflamingo_3 --experiment adversarial --dataset sakura-animal --adversarial-aug concat --adversarial-variant wrong
python main.py --model qwen_omni_2.5 --experiment partial_filler_text --dataset mmar --partial-filler-mode random --filler-type lorem
```

Notes:

- `baseline` is the main prerequisite for dependent CoT interventions
- `jasco_masking` is a dedicated open-ended pipeline and should be treated separately from the standard multiple-choice workflow
- many runs support restartability, restricted subsets, and chunked/parallel execution

### 5. Run analysis

Active analysis lives under `analysis/`.

Start here:

- [analysis/README.md](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/analysis/README.md)
- [analysis/cot/README.md](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/analysis/cot/README.md)
- [analysis/audio_interventions/README.md](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/analysis/audio_interventions/README.md)

Examples:

```bash
python analysis/cot/plot_partial_filler.py --model qwen_omni_2.5 --mode random --filler-type lorem --save-pdf
python analysis/audio_interventions/jasco/jasco_evaluation_llm_as_a_judge.py --model qwen_omni_2.5 --judge mistral
python analysis/audio_interventions/jasco/evaluate_jasco_results.py --model qwen_omni_2.5 --judge mistral --save-pdf
```

### 6. View the companion website

You can inspect the local website directly from:

- `index.html`

The website mirrors the paper at a higher level and is intended as a reader-friendly front door for the benchmark, plots, and overall framing.

## Important Notes

- This repository is under active refactor toward cleaner public release and stronger reproducibility defaults.
- Naming conventions, shell helpers, and some folder-level documentation are being standardized incrementally.
- We prioritize restartability and HPC-friendly execution, so several workflows are designed to resume from partial results rather than rerun from scratch.
- Some model/data setup helpers are still being moved into the `shell_scripts/` layout; the root README will continue to evolve as that work lands.

## Pointers to Related Files

- JASCO dataset notes: [data/jasco/README.md](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/data/jasco/README.md)
- Shell-script layout: [shell_scripts/README.md](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/shell_scripts/README.md)
- Analysis layout: [analysis/README.md](/scratch/lovenya/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/analysis/README.md)
- Paper source: `LALM_Faithfulness/`

## Citation

Citation details will be added once the paper is public.

For the blind-review phase, please use the anonymous companion website and code snapshot links above.
