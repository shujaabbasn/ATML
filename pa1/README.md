# PA1

## Task 1: Inductive Biases and Feature Representations

Run `task1/task1_inductive_biases.ipynb` top to bottom on a Colab T4 (~25 min). Cell 1 mounts
Drive and sets `PROJECT_DIR`. STL-10 downloads on first run into `data/`, which is untracked.

Seed 6304 throughout. Environment in `task1/requirements.txt`.

Outputs are in `task1/results/`: `subset.json` (split and the frozen 500-image eval subset),
`task1_results.json` (all reported numbers), `figures/`.

- AdaIN cue conflicts use the architecture and released weights of
  https://github.com/naoto0804/pytorch-AdaIN (Huang and Belongie, 2017).
- Backbones: torchvision ResNet-50 and ViT-B/16, OpenCLIP ViT-B-32-quickgelu.
