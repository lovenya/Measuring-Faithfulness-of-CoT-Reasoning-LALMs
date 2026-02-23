# TODO

## 1. SNR Robustness Experiment

- [ ] Generate noisy audio data (request more CPU cores for speed)
  - [ ] MMAR: `python data_processing/generate_noisy_audio.py --source data/mmar --output data/mmar_noisy`
  - [ ] Sakura (all 4 tracks): `python data_processing/generate_noisy_audio.py --source data/sakura --output data/sakura_noisy`
- [ ] Verify noisy data: correct file counts, audio loadable, SNR levels correct
- [ ] Demo run experiment on 2 samples: `python main.py --model qwen --experiment snr_robustness --dataset mmar --num-samples 2`
- [ ] Create sbatch submission scripts (4 models × 5 datasets = 20 scripts)
  - Models: qwen, salmonn, salmonn_7b, flamingo
  - Datasets: mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language
- [ ] Submit all jobs
- [ ] Run analysis: `python analysis/evaluate_snr_robustness.py --model qwen`
- [ ] Create plotting script (`analysis/per_dataset/plot_snr_robustness.py`)

## 2. JASCO Experiment

- [x] Fix newline bug in `jasco_masking.py`
- [x] Rewrite `evaluate_jasco.py` (--model arg, last-line parsing, tqdm, --judge CLI)
- [x] Run Stage 1 (Qwen) — 1,680 entries generated ✓
- [x] Run Stage 2 evaluation (Mistral judge) — scored ✓
- [ ] Run JASCO for SALMONN 13B, SALMONN 7B, Flamingo (Stage 1 + Stage 2)
- [ ] Create JASCO plotting/analysis script

## 3. Parallelization Pipeline Improvements

- [ ] Create `data_processing/verify_parallel_completeness.py`
  - Check entry counts per chunk vs expected (e.g., 11 entries/sample for audio_masking)
  - Show per-chunk PASS/FAIL with expected vs actual
- [ ] Update `data_processing/merge_parallel_results.py`
  - Refuse to merge if any chunk is incomplete
  - Add `--force-merge` CLI flag to override
  - Add `--expected-entries-per-sample` flag
  - Print summary showing which chunks passed/failed
