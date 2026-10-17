# ATML PA2 - LLM Post-Training

Syed Shuja Abbas Naqvi (28100250). Run everything from this `pa2/` folder. All runs used 1 Kaggle T4 and seed 6304. Hyperparameters are in `configs/` and every result is in `results/`.

## Setup

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

## Commands

```bash
# Task 1 - DPO
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml

# Task 2 - PPO
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

# Task 3 - GRPO
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml

# Task 4 - Safety (needs the Task 1-3 standard adapters in outputs/)
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml

# Task 5 - RLVR vs RLAIF
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

The Task 4 manual labels were filled by hand in `results/task4_safety/manual_audit_sheet.csv` before looking at the judge labels. To reproduce, skip `make_audit_sheet` since it overwrites that sheet with empty labels.

`kaggle_runs/` holds the notebooks that launched these commands.

## Attribution

Built on the course starter code (https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining) and assets (https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets). I fixed the planted defects in `task1_dpo/dpo.py`, `task2_ppo/ppo.py` and `task3_grpo/grpo.py` and wrote the training, ablation and evaluation code. No other external code was reused.
