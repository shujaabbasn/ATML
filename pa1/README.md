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

## Task 2: Unsupervised Domain Adaptation

Run `task2/task2_domain_adaptation.ipynb` top to bottom on a Colab T4 (~1 h). Cell 1 mounts Drive and sets `PROJECT_DIR`, which must be the folder that contains `task2/` and `shared/`. PACS (`flwrlabs/pacs` on Hugging Face) downloads on first run into `hf_cache/`, which is untracked. Sketch is the unlabelled target; Photo, Art Painting and Cartoon are the labelled sources, each split 80/20 (stratified) into training and validation.
Seed 6304 throughout. Environment in `task2/requirements.txt`. Seed, domains, split and image sizes are in `task2/configs/task2.yaml`; optimizer and schedule (AdamW, learning rate 1e-4, weight decay 1e-4, at most 30 epochs, patience 5 on mean source-validation macro-F1) are set in cell 7. Every update uses 8 examples from each source domain plus 24 target images and BatchNorm running statistics stay frozen at their ImageNet values.
Outputs are in `task2/results/`: `task2_results.json` (all reported numbers: per-domain and target metrics, domain separability, per-class target accuracy, the λ_MMD study, hyperparameters and training histories), `figures/`, and `checkpoints/source_only.pt`, which is reused unchanged as the Task 3 ERM baseline. The split protocol shared by Tasks 2 and 3 is `shared/splits/pacs_sketch_seed6304.json`. Re-running retrains everything and overwrites the checkpoints.

* Source-only, DAN (multi-kernel MMD, λ_MMD = 1), DANN (gradient reversal, α(p) = 2/(1+exp(−10p)) − 1) and CDAN (discriminator on f ⊗ p, no entropy conditioning) are implemented in the notebook following Ganin et al. (2016).
* Backbone: torchvision ResNet-18 (`IMAGENET1K_V1`), fine-tuned end to end. Dataset: PACS
* `dann_strength_0.0` in the histories is a control run with gradient reversal switched off, used to check that the training pipeline itself is correct.
* Determinism flags are set (cuDNN, cuBLAS workspace, deterministic algorithms in warn-only mode), but GPU runs are not guaranteed to be bit-identical. Reported numbers come from a single run of the notebook.
