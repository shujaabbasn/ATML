# Kaggle runs

These notebooks are only launchers. Every cell runs one `python -m ...` command from this repository, so the
experiment code lives in the task folders, not here. Each notebook is one Kaggle session (GPU T4, internet on).
Push `pa3/` to GitHub first, the first cell clones the repo.

| Notebook | Session | Results |
|---|---|---|
| `task1_full_ft_run.ipynb` | Full FT, then baseline / midpoint / final evaluation and qualitative | `results/task1/` |
| `task1_l2sp_run.ipynb` | Full FT + L2-SP, evaluation and qualitative | `results/task1/` |
| `task1_adapters_run.ipynb` | LoRA-4 and LoRA-16 in parallel (T4 x2), then OPLoRA-16, merges, evaluation, qualitative | `results/task1/` |
| `task1_analysis_run.ipynb` | Copies the three notebooks' outputs in, then qualitative collection and diagnostics | `results/task1/` |
| `task2_vlm_run.ipynb` | Data, SigLIP feature cache, three-stage training, final evaluation (+ optional 49 tokens) | `results/task2/` |
| `task3_ttc_run.ipynb` | Supplied rollouts: preparation, offline selection, adaptive stopping, live generation, examples | `results/task3/` |

Each full fine-tuning run gets its own notebook because Kaggle keeps at most 20GB in `/kaggle/working`
(fp32 model + AdamW recovery checkpoint is about 7GB, plus the midpoint and final weights).

For `task1_analysis_run.ipynb`, add the outputs of the first three notebooks as inputs and set their folder names in the first cell.

If a session dies, rerun the same training command with `--resume`, it continues from the newest recovery checkpoint
(only if that session's output was saved).
