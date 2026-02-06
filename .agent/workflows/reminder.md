---
description: Daily reminder/todo tracker - call at start of each coding session
---

# /reminder Workflow

This workflow shows the current status of ongoing work and pending tasks.

## Steps

1. **Read the current task.md artifact** to see what's in progress and what's remaining.

2. **Check for any running jobs** using:

   ```bash
   squeue -u $USER -o "%.10i %.40j %.8T %.10M %.6D %R"
   ```

3. **Review recent results** - check if any experiments completed since last session.

4. **Summarize status** - provide a brief update on:

   - What was completed in the last session
   - What's currently in progress (running jobs)
   - What's next to do

5. **Ask the user** what they want to focus on for this session.

---

## Current Focus Areas (Dec 2025)

### Phase 1: Start/End Filler Tokens

- `partial_filler_text` and `flipped_partial_filler_text`
- Models: Flamingo, SALMONN 7B, SALMONN 13B

### Phase 2: Lorem Ipsum Filler

- `random_partial_filler_text --filler-type lorem`
- Models: Flamingo, SALMONN 7B

### Phase 3: Mistral Perturbations

- `adding_mistakes --perturbation-source mistral`
- `paraphrasing --perturbation-source mistral`
- Models: Flamingo, SALMONN 7B

### Completed

- Extended `correct_vs_incorrect_analysis.py` for Flamingo/SALMONN 7B
- README website link update
