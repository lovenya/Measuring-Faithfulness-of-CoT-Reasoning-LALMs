# Experiment Status & TODO

## Summary
Testing Mistral perturbation experiments on Rorqual cluster.
**Last Updated:** 2025-12-19 17:17

---

## 🔄 CURRENTLY RUNNING

### Adding Mistakes Experiments
| Model | Dataset | Job ID | Account | Status |
|-------|---------|--------|---------|--------|
| SALMONN | mmar | 4467910 | rrg-csubakan | 🔄 Running |
| SALMONN | sakura-animal | 4467911 | rrg-csubakan | 🔄 Running |
| SALMONN | sakura-emotion | 4467912 | rrg-csubakan | 🔄 Running |
| SALMONN | sakura-gender | 4467913 | rrg-csubakan | 🔄 Running |
| SALMONN | sakura-language | 4467914 | rrg-csubakan | 🔄 Running |
| Qwen | mmar | 4467945 | rrg-ravanelm | 🔄 Running |
| Qwen | sakura-animal | 4468335 | rrg-ravanelm | 🔄 Running |
| Qwen | sakura-emotion | 4468336 | rrg-ravanelm | 🔄 Running |
| Qwen | sakura-gender | 4468337 | rrg-ravanelm | 🔄 Running |
| Qwen | sakura-language | 4468338 | rrg-ravanelm | 🔄 Running |

### Mistral Perturbation Generation
| Model | Dataset | Type | Job ID | Status |
|-------|---------|------|--------|--------|
| Qwen | sakura-emotion | Paraphrase | 4467939 | 🔄 Running |
| Qwen | sakura-language | Paraphrase | 4467938 | 🔄 Running |
| Qwen | sakura-animal | Paraphrase | 4468365 | 🔄 Running |
| Qwen | mmar | Paraphrase | 4468366 | 🔄 Running |

---

## ✅ COMPLETED

### Perturbation Generation (Mistral)
| Model | Dataset | Mistakes | Paraphrases |
|-------|---------|----------|-------------|
| SALMONN | mmar | ✅ | ✅ |
| SALMONN | sakura-animal | ✅ | ✅ |
| SALMONN | sakura-emotion | ✅ | ✅ |
| SALMONN | sakura-gender | ✅ | ✅ |
| SALMONN | sakura-language | ✅ | ✅ |
| Qwen | mmar | ✅ | 🔄 Running |
| Qwen | sakura-animal | ✅ | 🔄 Running |
| Qwen | sakura-emotion | ✅ | 🔄 Running |
| Qwen | sakura-gender | ✅ | ✅ |
| Qwen | sakura-language | ✅ | 🔄 Running |

---

## ❌ NOT STARTED

### SALMONN Paraphrasing Experiments
| Dataset | Status |
|---------|--------|
| mmar | ❌ Needs demo first |
| sakura-animal | ❌ Needs demo first |
| sakura-emotion | ❌ Not started |
| sakura-gender | ❌ Not started |
| sakura-language | ❌ Not started |

### Qwen Paraphrasing Experiments
| Dataset | Status |
|---------|--------|
| mmar | ❌ Not started |
| sakura-animal | ❌ Not started |
| sakura-emotion | ❌ Not started |
| sakura-gender | ❌ Not started |
| sakura-language | ❌ Not started |

---

## Environment Setup Status

| Environment | Status |
|-------------|--------|
| `qwen_env` | ✅ Ready |
| `salmonn_env` | ✅ Ready |
| `mistral_env` | ✅ Ready |

### Offline Mode Fixes Applied
- ✅ Downloaded `bert-base-uncased` to `model_components/`
- ✅ Added `local_files_only=True` to all `from_pretrained()` calls in `salmonn.py`
- ✅ Environment vars: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
- ✅ Fixed `_mistakesd` typo to `_mistakes` in 12 submission scripts
