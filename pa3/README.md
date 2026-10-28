# ATML PA3 - Fine-Tuning, Visual Grounding and Test-Time Compute

Syed Shuja Abbas Naqvi (28100250). Run everything from this `pa3/` folder. Seed 6304 everywhere. Settings are in `configs/`, small machine-readable results go to `results/`, model weights and caches go to `outputs/` and `data/` (both gitignored).

## Setup

```bash
python -m pip install -r requirements.txt
```

## Task 1 - LLM fine-tuning (Qwen3-0.6B on MetaMathQA)

```bash
python -m task1_finetune.prepare_data
python -m task1_finetune.train --run full_ft
python -m task1_finetune.train --run full_ft_l2sp
python -m task1_finetune.train --run lora4
python -m task1_finetune.train --run lora16
python -m task1_finetune.train --run oplora16
python -m task1_finetune.merge --run lora4
python -m task1_finetune.merge --run lora16
python -m task1_finetune.merge --run oplora16
python -m task1_finetune.evaluate --name baseline   # also full_ft_mid, full_ft, full_ft_l2sp, lora4, lora16, oplora16
python -m task1_finetune.qualitative --name baseline   # same seven names, then:
python -m task1_finetune.qualitative --collect
python -m task1_finetune.diagnostics
```

`train` takes `--resume` to continue from the newest recovery checkpoint (updates 312, 625, 937, 1250).

## Task 2 - SigLIP-Qwen VLM and visual grounding

```bash
python -m task2_vlm.prepare_data
python -m task2_vlm.cache_features
python -m task2_vlm.train
python -m task2_vlm.evaluate
python -m task2_vlm.train --visual-tokens 49      # optional step 7
python -m task2_vlm.evaluate --visual-tokens 49   # optional step 7
```

`train` takes `--resume` to continue from the newest checkpoint (every 250 updates).

## Task 3 - Test-time compute

```bash
python -m task3_ttc.prepare
python -m task3_ttc.offline_eval --dataset math
python -m task3_ttc.offline_eval --dataset mathvista
python -m task3_ttc.adaptive --dataset math
python -m task3_ttc.adaptive --dataset mathvista
python -m task3_ttc.live_generation
python -m task3_ttc.examples
```

`live_generation` appends one line per question and skips finished questions when rerun.

## Attribution

- `task2_vlm/vqa_eval.py`: answer normalization ported from the official VQA evaluation code (https://github.com/GT-Vision-Lab/VQA).
- `task3_ttc/answers.py`: MathVista answer normalization follows `normalize_extracted_answer` in https://github.com/lupantech/MathVista, changed to keep multiple-choice answers as option letters.
- Benchmarks run through lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness). Qualitative prompts come from `first_plot_questions.yaml` in https://github.com/emergent-misalignment/emergent-misalignment.
- Task 3 data: the course's ATML-PA3-Data folder. Equivalence checks use math-verify (https://github.com/huggingface/Math-Verify).

No other external code was reused.
