## High-level refactor checklist

- **Model aliases**
  - [x] Introduce public aliases: `qwen_audio_2`, `qwen_omni_2.5`, `aflamingo_3`, `salmonn_audio_7b`, `salmonn_audio_13b`.
  - [ ] Sweep experiments, analysis, and scripts to ensure they only use these (keep legacy aliases temporarily for back-compat).

- **External perturbations (paraphrasing / adding_mistakes)**
  - [x] Make output filename suffix depend on `EXTERNAL_LLM` instead of hardcoding `-mistral`.
  - [ ] Thread a clean `perturbation_source` / external-LLM name through experiments, data_processing, and analysis so naming is consistent everywhere.

- **Main orchestrator (`main.py`)**
  - [x] Set default filler type to `lorem` (both CLI and `config.FILTER_TYPE`).
  - [x] Remove `--temperature/--top-p/--top-k` from the CLI and rely on model defaults.
  - [ ] Refactor independent experiment handling so they no longer depend on baseline at run time (only analysis scripts compare to baseline).
  - [ ] Refactor dependent experiment handling so they don’t depend on `no_reasoning` at run time; `no_reasoning` remains foundational only.
  - [ ] Consider splitting `main.py` into smaller modules (CLI definition, path management, experiment loading) once all experiments are stable.

- **Experiments**
  - [ ] Foundational: tidy `baseline` and `no_reasoning` scripts, keeping behavior identical but improving structure and comments.
  - [ ] Dependent CoT interventions: remove any hard dependency on `no_reasoning` outputs; rely on `baseline` only, and let analysis handle `no_reasoning` comparisons.
  - [ ] Filler text family: consolidate `filler_text`, `partial_filler_text`, `flipped_partial_filler_text`, and `random_partial_filler_text` into a single parameterized script with a clear `mode` flag.
  - [ ] Audio interventions: rename `audio_masking` to `partial_audio_masking` across code and analysis (with aliases for back-compat), and remove baseline/no_reasoning coupling from the experiment layer.

- **Flamingo utils**
  - [x] Standardize on the HF-backed `audio_flamingo_hf_utils` for `aflamingo_3`.
  - [ ] Remove the old `audio_flamingo_utils` file and any dead references once we’re confident no job scripts use it.

- **Analysis / plotting**
  - [ ] Prune analysis scripts that are no longer needed or are superseded by cleaner cross-dataset plots.
  - [ ] Make all consistency / accuracy comparisons live entirely in analysis (experiments just write per-trial JSONL with minimal assumptions).
  - [ ] Ensure analysis scripts that compute *baseline consistency* load `baseline` JSONLs directly (no experiment-level dependency on `no_reasoning`), and document this clearly.
  - [ ] Ensure analysis understands the new alias and external-perturbation naming scheme.

- **Reproducibility & scripts**
  - [ ] Add shell scripts to create per-model Python environments (`qwen_audio_2`, `qwen_omni_2.5`, `aflamingo_3`, `salmonn_audio_7b/13b`).
  - [ ] Add scripts to download models and datasets to user-specified locations and update `config` (or a local override) automatically.
  - [ ] Add gentle runtime warnings when configured model/dataset paths do not exist, prompting users to run the setup scripts.

- **Documentation**
  - [ ] Rebuild `README.md` from scratch once core aliases, experiments, and scripts are stable.
  - [ ] Document the offline/HPC-first workflow, including local weights, Slurm/array usage, and restartable behavior.

