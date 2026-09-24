# PA1: Beyond IID

This folder has my code and results for all four tasks of PA1. Each task is one Colab notebook.
All of them use seed 6304 and were run on Colab.

## How to run

1. Put this `pa1` folder in Google Drive at `MyDrive/ATML-PA1` (or change `PROJECT_DIR` in the
   first cell of each notebook).
2. Open a notebook in Colab and run it from top to bottom. The first cell mounts Drive.
3. Run Task 2 before Task 3, because Task 3 uses the Source-only model that Task 2 trains.

Datasets download automatically the first time (STL-10, PACS from Hugging Face, CIFAR-10 and
CIFAR-100). Model checkpoints (`*.pt`) are not in the repo because they are about 400 MB in total.
The notebooks save them to `<task>/results/checkpoints/` when they run. Every number in my report
comes from the `*_results.json` file of its task.

Each task folder has:
- the notebook
- `configs/` with the main settings
- `requirements.txt` with the package versions
- `results/` with the results JSON and figures

## Task 1: Inductive biases (`task1/`)

It compares a ResNet-50, a ViT-B/16 and CLIP on STL-10 under different image
changes:
- grayscale
- a colour swap from another class
- shape/texture cue conflicts made with AdaIN
- translation
- patch shuffling

`results/subset.json` has the train/validation split and the 500 test images used for evaluation.

## Task 2: Domain adaptation (`task2/`)

It trains on PACS Photo, Art Painting and Cartoon and adapts to Sketch without Sketch
labels. It compares four methods: Source-only, DAN, DANN and CDAN. The PACS split is saved in
`shared/splits/pacs_sketch_seed6304.json` so that Task 3 uses exactly the same split.

If the Source-only and DAN checkpoints are already saved, the notebook loads them instead of
training them again. That way Task 3 always gets the same Source-only model.

### Changes I made to the manual's setup

Some methods would not train properly as written, so I made three changes. The TA allowed this if it
is documented. Each change is applied the same way to every method it affects. I decided on all
three by looking only at the training losses and source validation scores, never at Sketch results.

1. **Unbiased MMD.** The normal (biased) MMD formula is always a bit above zero on small batches,
   even when the two batches come from the same distribution. With a strong weight this made
   DAN (and DAN-DG in Task 3) predict one class for everything (source validation F1 of 0.051). I
   switched to the unbiased version, which leaves out each sample's comparison with itself.
2. **Gradient clipping** at norm 1.0 for every method in Tasks 2 and 3. It helped, but DANN still
   blew up (gradient norm around 12 million).
3. **Normalised discriminator input for DANN and CDAN.** DANN could fool its discriminator just by
   making the features bigger and bigger. I scale the features to length 1 before they go into
   the discriminator only. The classifier still sees the normal features. After this, DANN and
   CDAN trained stably.

`results/task2_results_before_unit_length.json` is the run before change 3, which shows the
blow-up.

## Task 3: Domain generalization (`task3/`)

It uses the same PACS split as Task 2, but Sketch is never seen during training or
model selection. The code throws an error if a Sketch image reaches any training code, and only
the last evaluation cell loads Sketch. It compares:
- ERM, which is Task 2's Source-only model, loaded and not retrained
- DAN-DG, which aligns the three source domains with each other
- SAM (sharpness-aware minimization)

It uses the first two changes from Task 2.

## Task 4: Open-set recognition (`task4/`)

CIFAR-10 is the known set. Eight "near" and eight "far" CIFAR-100 test classes are the unknowns. It
compares a normal model (Vanilla), the same model trained with RandAugment (GCSC) and PROSER.

- Cells 4 to 6 train the models.
- Cells 7 to 15 compute all the results from the saved checkpoints.
- CIFAR-100 images stay locked in the code until all three models are trained. They are never used
  for training or for picking thresholds.

## Code I reused

- AdaIN style transfer (Task 1) uses the pretrained decoder and VGG weights from
  [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN). The network layout matches
  that repo so the weights load.
- The PROSER unknown score (Task 4) follows the official code of Zhou et al. (2021), including its
  temperature of 1024.
- The backbones come from torchvision and OpenCLIP.
