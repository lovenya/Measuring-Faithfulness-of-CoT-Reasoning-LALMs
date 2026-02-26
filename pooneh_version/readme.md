# Audio Faithfulness Experiments — Model Runner

This repository provides a unified interface to run **audio faithfulness and intervention experiments** across multiple large audio-language models (e.g., Qwen-Instruct, Qwen2.5-Omni, AFthink / Audio-Flamingo 3, SALMONN).  
Each model is wrapped with a standardized runner so experiments can be reproduced consistently.

[Image of machine learning inference pipeline showing standard zero-shot generation versus reasoning-conditioned generation]

---

## Overview

The pipeline follows four main steps:

1. **Environment Setup:** Create an environment and install model-specific dependencies.
2. **Data Preparation:** Download the Sakura dataset repository.
3. **Baseline Inference:** Run the model wrapper to generate a reasoning chain and initial prediction.
4. **Conditioned Inference (Faithfulness Test):** Re-run the model, forcing it to use its own generated reasoning to see if it remains faithful to its logic.

---

## Step 1 — Environment Setup & Dependencies

We strongly recommend creating a **separate Conda environment per model** to avoid severe dependency conflicts (especially between different versions of `transformers`, `peft`, and `accelerate`).

Please install dependencies by following each model’s official page:

* **AF3 (Audio Flamingo 3):** [https://huggingface.co/nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf)
* **Qwen2.5-Omni:** [https://huggingface.co/Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
* **SALMONN / Qwen-Instruct:** Follow their respective official repository guidelines.

*Example:*
```bash
conda create -n af3-env python=3.10 -y
conda activate af3-env
# Install specific requirements here...
```

## Step 2 — Get Sakura Repository & Prepare Data
Clone the Sakura repository and ensure the audio data is saved locally. You will need to pass the path to this data when running the inference scripts.

```bash
mkdir -p data/
cd data/
git clone [https://github.com/ckyang1124/SAKURA.git](https://github.com/ckyang1124/SAKURA.git)
```

## Step 3 — Run Baseline Inference

Use the provided wrapper scripts (e.g., AF3_wrapper.py) to execute the baseline model runs. These scripts prompt the model to generate a step-by-step reasoning chain followed by a final multiple-choice letter prediction.

Required Arguments:

--input: Path to the .jsonl manifest containing instructions and audio paths.

--output: Destination path for the generated predictions and reasoning.

--data_root: The path to the root directory where the SAKURA repository was cloned.

Command Example:
```bash
python interface/AF3_wrapper.py \
  --input processed_input/sakura/{name}/sakura_manifest.jsonl \
  --output output/sakura/af3/{name}/baseline_{name}.jsonl \
  --data_root data/
  ```

## Step 4 — Run Conditioned Inference (Faithfulness)
To test if the model actually relies on its own logic, use the conditioned inference scripts. These scripts take the runs_reasonings generated in Step 3, inject them back into the prompt, and force the model to make a final prediction based strictly on that text.

Command Example:
```bash 
  python interface/{model}_conditioned.py \
  --manifest_file  processed_input/sakura/{name}/sakura_manifest.jsonl \
  --input_results_file output/sakura/af3/{name}/baseline_{name}.jsonl \
  --output_conditioned_file output/sakura/af3/{name}/conditioned_{name}.jsonl \
  --data_root data/
  ```
