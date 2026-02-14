# Experiments Status Tracker

> Last updated: 2026-02-13

---

## ✅ Completed Experiments

### Qwen — Audio Masking (start + end modes)

These experiments exist in `results/qwen/audio_masking/{silence,noise}/{start,end}/` but most did **not finish** within
their original 15h time limit. They will be **re-submitted** with increased times (see Ready to Submit below).

---

## 🟡 Ready to Submit (Scripts Created, NOT Yet Submitted)

### Qwen — Audio Masking

**Resources (all jobs):** `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1`, `--cpus-per-task=3`, `--mem=64G`, `--account=rrg-ravanelm`

> [!IMPORTANT]
> **Scattered mode** requires generating masked audio first via `sbatch submission_scripts/data_processing/generate_masked_datasets.sh`
> Start/end modes already have masked audio data generated.

#### Noise + End

| Dataset         | Time | Script                                                                        |
| --------------- | ---- | ----------------------------------------------------------------------------- |
| mmar            | 15h  | `submission_scripts/qwen/audio_masking/noise/end/run_qwen_mmar.sh`            |
| sakura-animal   | 30h  | `submission_scripts/qwen/audio_masking/noise/end/run_qwen_sakura_animal.sh`   |
| sakura-emotion  | 30h  | `submission_scripts/qwen/audio_masking/noise/end/run_qwen_sakura_emotion.sh`  |
| sakura-gender   | 40h  | `submission_scripts/qwen/audio_masking/noise/end/run_qwen_sakura_gender.sh`   |
| sakura-language | 40h  | `submission_scripts/qwen/audio_masking/noise/end/run_qwen_sakura_language.sh` |

#### Noise + Scattered

| Dataset         | Time | Script                                                                              |
| --------------- | ---- | ----------------------------------------------------------------------------------- |
| mmar            | 25h  | `submission_scripts/qwen/audio_masking/noise/scattered/run_qwen_mmar.sh`            |
| sakura-animal   | 25h  | `submission_scripts/qwen/audio_masking/noise/scattered/run_qwen_sakura_animal.sh`   |
| sakura-emotion  | 25h  | `submission_scripts/qwen/audio_masking/noise/scattered/run_qwen_sakura_emotion.sh`  |
| sakura-gender   | 25h  | `submission_scripts/qwen/audio_masking/noise/scattered/run_qwen_sakura_gender.sh`   |
| sakura-language | 25h  | `submission_scripts/qwen/audio_masking/noise/scattered/run_qwen_sakura_language.sh` |

#### Noise + Start

| Dataset         | Time | Script                                                                          |
| --------------- | ---- | ------------------------------------------------------------------------------- |
| mmar            | 10h  | `submission_scripts/qwen/audio_masking/noise/start/run_qwen_mmar.sh`            |
| sakura-animal   | 30h  | `submission_scripts/qwen/audio_masking/noise/start/run_qwen_sakura_animal.sh`   |
| sakura-emotion  | 30h  | `submission_scripts/qwen/audio_masking/noise/start/run_qwen_sakura_emotion.sh`  |
| sakura-gender   | 8h   | `submission_scripts/qwen/audio_masking/noise/start/run_qwen_sakura_gender.sh`   |
| sakura-language | 20h  | `submission_scripts/qwen/audio_masking/noise/start/run_qwen_sakura_language.sh` |

#### Silence + End

| Dataset         | Time | Script                                                                          |
| --------------- | ---- | ------------------------------------------------------------------------------- |
| mmar            | 10h  | `submission_scripts/qwen/audio_masking/silence/end/run_qwen_mmar.sh`            |
| sakura-animal   | 20h  | `submission_scripts/qwen/audio_masking/silence/end/run_qwen_sakura_animal.sh`   |
| sakura-emotion  | 20h  | `submission_scripts/qwen/audio_masking/silence/end/run_qwen_sakura_emotion.sh`  |
| sakura-gender   | 12h  | `submission_scripts/qwen/audio_masking/silence/end/run_qwen_sakura_gender.sh`   |
| sakura-language | 20h  | `submission_scripts/qwen/audio_masking/silence/end/run_qwen_sakura_language.sh` |

#### Silence + Scattered

| Dataset         | Time | Script                                                                                |
| --------------- | ---- | ------------------------------------------------------------------------------------- |
| mmar            | 25h  | `submission_scripts/qwen/audio_masking/silence/scattered/run_qwen_mmar.sh`            |
| sakura-animal   | 25h  | `submission_scripts/qwen/audio_masking/silence/scattered/run_qwen_sakura_animal.sh`   |
| sakura-emotion  | 25h  | `submission_scripts/qwen/audio_masking/silence/scattered/run_qwen_sakura_emotion.sh`  |
| sakura-gender   | 25h  | `submission_scripts/qwen/audio_masking/silence/scattered/run_qwen_sakura_gender.sh`   |
| sakura-language | 25h  | `submission_scripts/qwen/audio_masking/silence/scattered/run_qwen_sakura_language.sh` |

#### Silence + Start

| Dataset         | Time | Script                                                                            |
| --------------- | ---- | --------------------------------------------------------------------------------- |
| mmar            | 10h  | `submission_scripts/qwen/audio_masking/silence/start/run_qwen_mmar.sh`            |
| sakura-animal   | 20h  | `submission_scripts/qwen/audio_masking/silence/start/run_qwen_sakura_animal.sh`   |
| sakura-emotion  | 20h  | `submission_scripts/qwen/audio_masking/silence/start/run_qwen_sakura_emotion.sh`  |
| sakura-gender   | 12h  | `submission_scripts/qwen/audio_masking/silence/start/run_qwen_sakura_gender.sh`   |
| sakura-language | 20h  | `submission_scripts/qwen/audio_masking/silence/start/run_qwen_sakura_language.sh` |

**Total: 30 jobs**

---

## 🔲 In Pipeline (Not Yet Implemented)

- [ ] SALMONN audio masking experiments (all mask types and modes)
- [ ] Audio Flamingo audio masking experiments
- [ ] Analysis and final plots for all audio masking results
