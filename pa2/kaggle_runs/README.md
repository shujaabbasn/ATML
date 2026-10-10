# Kaggle runs

These notebooks are only launchers. Every cell runs one `python -m ...` command from this repository, so the
experiment code lives in the task folders, not here. They are kept as a record of the exact runs behind the
results in `results/` (GPU, package versions, printed tables and timings).

All runs: Kaggle, one Tesla T4 (`CUDA_VISIBLE_DEVICES=0`), Transformers 4.57.1, PEFT 0.17.1, TRL 0.27.2.

| Notebook | Task | Results |
|---|---|---|
| `task1_dpo_run.ipynb` | Task 1, DPO | `results/task1_dpo/` |
| `task2_ppo_run.ipynb` | Task 2, PPO | `results/task2_ppo/` |
| `task3_grpo_run.ipynb` | Task 3, GRPO | `results/task3_grpo/` |
| `task4_safety_run.ipynb` | Task 4, safety calibration | `results/task4_safety/` |
