# Job Tracking Document - December 20, 2025

Last updated: Sat Dec 20 10:42:00 EST 2025

---

## Summary of Experiment Status

| Experiment | Qwen | SALMONN |
|------------|------|---------|
| Adding Mistakes (Mistral) | ✅ All 5 DONE | ⏳ 3 pending (mmar, animal, emotion) |
| Paraphrasing (Mistral) | ✅ All 5 DONE | ✅ All 5 DONE |
| Lorem Filler (random_partial) | ✅ All 5 DONE | ✅ All 5 DONE |

---

## Currently Running/Pending Jobs

### SALMONN Adding Mistakes - Continuation Jobs (rrg-ravanelm, 40GB GPU)

| Job ID | Dataset | Time | Status | Output File |
|--------|---------|------|--------|-------------|
| **4489048** | mmar | 8h | PENDING | `adding_mistakes_salmonn_mmar-restricted-mistral.jsonl` |
| **4489049** | sakura-animal | 14h | PENDING | `adding_mistakes_salmonn_sakura-animal-restricted-mistral.jsonl` |
| **4489050** | sakura-emotion | 12h | PENDING | `adding_mistakes_salmonn_sakura-emotion-restricted-mistral.jsonl` |

**Log locations:** `logs/experiment_mistral/`

---

## Completed Experiments

### ✅ Qwen Adding Mistakes (Mistral) - ALL DONE
- mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language
- Output: `results/qwen/adding_mistakes/adding_mistakes_qwen_<dataset>-restricted-mistral.jsonl`

### ✅ SALMONN Adding Mistakes (Mistral) - 2/5 DONE
- ✅ sakura-gender, sakura-language
- ⏳ mmar, sakura-animal, sakura-emotion (continuation jobs submitted)
- Output: `results/salmonn/adding_mistakes/adding_mistakes_salmonn_<dataset>-restricted-mistral.jsonl`

### ✅ Qwen Paraphrasing (Mistral) - ALL DONE
- mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language
- Output: `results/qwen/paraphrasing/paraphrasing_qwen_<dataset>-restricted-mistral.jsonl`

### ✅ SALMONN Paraphrasing (Mistral) - ALL DONE
- mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language
- Output: `results/salmonn/paraphrasing/paraphrasing_salmonn_<dataset>-restricted-mistral.jsonl`

### ✅ Qwen Lorem Filler (random_partial_filler_text) - ALL DONE
- mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language
- Output: `results/qwen/random_partial_filler_text/random_partial_filler_text_qwen_<dataset>-restricted-lorem.jsonl`

### ✅ SALMONN Lorem Filler (random_partial_filler_text) - ALL DONE
- mmar, sakura-animal, sakura-emotion, sakura-gender, sakura-language
- Output: `results/salmonn/random_partial_filler_text/random_partial_filler_text_salmonn_<dataset>-restricted-lorem.jsonl`

---

## Jobs to Monitor

| Priority | Job ID | Description | Time Remaining (approx) |
|----------|--------|-------------|------------------------|
| 🔴 High | 4489048 | SALMONN mistakes mmar | ~8h |
| 🔴 High | 4489049 | SALMONN mistakes animal | ~14h |
| 🔴 High | 4489050 | SALMONN mistakes emotion | ~12h |

---

## Quick Commands

```bash
# Check job status
squeue -u lovenya

# Interactive node (3h, rrg-csubakan) - Run this command yourself:
salloc --time=3:00:00 --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 --cpus-per-task=8 --mem=64G --account=rrg-csubakan

# Check specific job logs
tail -f logs/experiment_mistral/salmonn-mistakes-*-cont-<JOB_ID>.err
```
