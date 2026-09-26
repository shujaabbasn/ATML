# PA1

Code and results for the four PA1 tasks. Each task is one Colab notebook and all experiments use
seed 6304.

## How to run

1. Put the `pa1` folder in Google Drive at `MyDrive/ATML-PA1` (or change `PROJECT_DIR` in the
   first cell of each notebook).
2. Run each notebook in Colab from top to bottom.
3. Run Task 2 before Task 3, since Task 3 reuses the Source-only model from Task 2.

Datasets download automatically. Checkpoints (`*.pt`, about 400 MB) are not in the repository and
are saved to `<task>/results/checkpoints/` when the notebooks run. Each task folder has the
notebook, `configs/`, `requirements.txt` and `results/` (results JSON and figures). Every number in
the report comes from the results JSON of its task.

Training ran on a Colab T4 GPU. The evaluation cells of Tasks 3 and 4 were last re-run on CPU from
the saved checkpoints, which gives the same numbers.

## Tasks

- **Task 1 (`task1/`)**: ResNet-50, ViT-B/16 and CLIP on STL-10 under grayscale, a colour swap,
  AdaIN cue conflicts, translation and patch shuffling. `results/subset.json` has the split and the
  500 test images.
- **Task 2 (`task2/`)**: Source-only, DAN, DANN and CDAN on PACS with Sketch as the unlabelled
  target. The split is saved in `shared/splits/pacs_sketch_seed6304.json`. Saved Source-only and
  DAN checkpoints are loaded if present, otherwise everything is trained from scratch.
- **Task 3 (`task3/`)**: ERM (the Task 2 Source-only model), DAN-DG and SAM on the same split.
  Sketch is only loaded in the final evaluation cell and the code raises an error if it reaches
  training.
- **Task 4 (`task4/`)**: Vanilla, GCSC and PROSER on CIFAR-10 with CIFAR-100 unknowns. Cells 4 to 6
  train and cells 7 to 15 evaluate. CIFAR-100 stays locked until all models are trained.

## Changes to the manual's setup

These were needed for training to work and were chosen from training losses and source validation
scores only, never from Sketch results.

1. **Unbiased MMD** (Tasks 2 and 3). The biased estimate collapsed DAN at weight 10 and DAN-DG at
   weight 1 to a single predicted class in earlier runs (kept in the repository history).
2. **Gradient clipping** at norm 1.0 (Tasks 2 and 3). This alone did not stop DANN from diverging.
3. **Unit-length discriminator input** for DANN and CDAN, since frozen BatchNorm let the features
   grow without limit. `task2/results/task2_results_before_unit_length.json` is the run before
   this change.

## Code I reused

- AdaIN decoder and VGG weights from
  [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).
- The PROSER unknown score follows the official code of Zhou et al. (2021).
- Backbones from torchvision and OpenCLIP.