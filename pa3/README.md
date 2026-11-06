# ATML PA3 - Fine-Tuning, Visual Grounding and Test-Time Compute

Syed Shuja Abbas Naqvi (28100250). Run everything from this `pa3/` folder. Seed 6304 everywhere.

- `configs/` settings, `results/` machine-readable results, split and example IDs
- `kaggle_runs/` the notebooks used to run this on a Kaggle T4
- `data/` and `outputs/` are downloaded data and model weights, not in the repo

```bash
python -m pip install -r requirements.txt
```

## Task 1 - LLM fine-tuning

```bash
python -m task1_finetune.prepare_data
python -m task1_finetune.train --run full_ft         # also full_ft_l2sp, lora4, lora16, oplora16
python -m task1_finetune.merge --run lora4           # also lora16, oplora16
python -m task1_finetune.evaluate --name baseline    # also full_ft_mid, full_ft, full_ft_l2sp, lora4, lora16, oplora16
python -m task1_finetune.qualitative --name baseline # same seven names
python -m task1_finetune.qualitative --collect
python -m task1_finetune.diagnostics
```

## Task 2 - SigLIP-Qwen VLM

```bash
python -m task2_vlm.prepare_data
python -m task2_vlm.cache_features
python -m task2_vlm.train                         # optional: --visual-tokens 49
python -m task2_vlm.evaluate                      # optional: --visual-tokens 49
python -m task2_vlm.examples
```

## Task 3 - Test-time compute

```bash
python -m task3_ttc.prepare
python -m task3_ttc.offline_eval --dataset math   # also mathvista
python -m task3_ttc.adaptive --dataset math       # also mathvista
python -m task3_ttc.live_generation
python -m task3_ttc.examples
```

## Attribution

- `task2_vlm/vqa_eval.py`: answer normalization from the official VQA evaluation code (https://github.com/GT-Vision-Lab/VQA).
- `task3_ttc/answers.py`: MathVista normalization follows https://github.com/lupantech/MathVista, keeping multiple-choice answers as letters.
- Benchmarks use lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness). Equivalence checks use math-verify (https://github.com/huggingface/Math-Verify).
- Qualitative prompts: `first_plot_questions.yaml` from https://github.com/emergent-misalignment/emergent-misalignment. Task 3 data: the course's ATML-PA3-Data folder.
