# Solar Filament Segmentation Challenge 2026 — Team Kaiser

Author: **Nguyen Van Son**.
Code, trained weights and technical report for **Team Kaiser** (Kaggle handle
[`snnguynvnk19hl`](https://www.kaggle.com/snnguynvnk19hl)) in the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup, MAGFiLO benchmark).

The repository contains a complete, self-contained segmentation pipeline: model, training code, inference code,
evaluation code and a notebook that runs the whole pipeline from the stored weights **without re-training**.

## Method in brief

| Stage | What we do |
|---|---|
| Input | 2048x2048 grayscale H-alpha image -> resized to 1024x1024 -> 3 channels in [0, 1]: raw intensity, CLAHE (clip 2.0, 8x8 tiles), Sobel gradient magnitude (divided by its maximum) |
| Network | ResNet-34 encoder (ImageNet-initialised) + transposed-convolution U-Net decoder, 24.45 M parameters, 1 output logit |
| Loss | 0.40 BCE + 0.35 soft Dice + 0.25 soft-skeleton clDice (Shit et al., CVPR 2021; 3 skeletonisation iterations) |
| Training data | Competition *train* split only, 566 images for training / 141 images for validation (split by image, sorted by file name, last 20 % held out); **7,522 training crops** of 512x512 centred on single annotated filaments (jitter, flips, 90-degree rotations) |
| Optimisation | AdamW, lr 3e-4, weight decay 1e-4, cosine schedule to 1e-6, batch 16, 18 epochs, mixed precision, seed 2026 |
| Inference | Full-disk image at 1024x1024, 4-view flip TTA (identity, H, V, HV) averaged, resized back to 2048x2048, threshold 0.98 (selected on the validation images) |
| Instances | 8-connected components of the binary mask (pixel-disjoint by construction); components < 1000 px discarded; optional area-guarded 3x3 cross opening (`--opening`) |
| Output | `filament_id = <image_stem>_<k>`, `segmentation_rle` = compressed COCO RLE counts (pycocotools) |

## Repository layout

```
model.py        ResNet-34 U-Net, 3-channel input representation (identical to training)
losses.py       BCE + Dice + soft-skeleton clDice loss
data.py         COCO annotation reading, train/val split, 512x512 crop dataset
train.py        training script (reproduces the released checkpoint's recipe)
pipeline.py     inference (TTA, instances, RLE), evaluation (Dice / PQ), submission validator
predict.py      CLI: test images -> submission.csv (+ validation of the file)
evaluate.py     CLI: Dice / PQ on the held-out validation images
make_report_figures.py   figures and bootstrap intervals of the technical report (figures/)
figure_pipeline.py       pipeline diagram (vector PDF) with thumbnails produced by the released model
reproduce_solar_filament_solution.ipynb   end-to-end notebook (loads weights, predicts, validates, evaluates)
models/best_model.pth                      trained weights (epoch 18, crop-level val Dice 0.7422)
technical_report.tex                       technical report (organizers' acmart sigconf template, 4-page limit)
requirements.txt                           pinned dependencies
```

## Reproduction

```bash
pip install -r requirements.txt
# put the competition data in data/MAGFiLO_1.0_Kaggle_2026/{train,test}  (or set FILAMENT_DATA_ROOT)

# 1) predictions for the test set (no training needed), ~0.6 s / image on an RTX 4060 laptop GPU
python predict.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images --output submission.csv

# 2) held-out validation (Dice, Panoptic Quality, fragmentation)
python evaluate.py --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
                   --images data/MAGFiLO_1.0_Kaggle_2026/train/train_images

# 3) or run everything in the notebook
jupyter notebook reproduce_solar_filament_solution.ipynb

# optional: re-train (about 1.5 h on a Tesla T4)
python train.py --ann <train.json> --images <train_images> --out runs/exp
```

`predict.py` finishes with a 10-point structural check of the CSV (columns, nulls, unique ids, id format, RLE
decoding at 2048x2048, empty masks, pixel overlap between instances, image coverage).

## Results

### Held-out validation of this repository's pipeline

141 validation images (677 ground-truth filaments) that were never used for training. Produced by
`evaluate.py`; PQ is our implementation of the published formula (IoU > 0.5, one-to-one matching, pooled over
images) and may differ in details from the official scorer.

Full-disk inference, 1024x1024 input, 4-view TTA, no opening. `thr` = probability threshold, `A` = minimum
instance area in pixels; PQ and Dice are point estimates with 95 % bootstrap intervals over the validation images
(2,000 resamples). The selected configuration (bold, also the default of `predict.py`) was chosen on these same
images, so its row is slightly optimistic; the best threshold lies at the edge of the explored grid.

| thr | A | PQ [95 % CI] | SQ | RQ | Dice [95 % CI] | TP / FP / FN | 1:N | N:1 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 60 | 0.020 [0.013, 0.028] | 0.593 | 0.034 | 0.199 [0.176, 0.224] | 29 / 1014 / 648 | 0 | 156 |
| 0.5 | 1000 | 0.022 [0.014, 0.030] | 0.593 | 0.037 | 0.200 [0.177, 0.226] | 29 / 854 / 648 | 0 | 156 |
| 0.8 | 60 | 0.056 [0.044, 0.069] | 0.597 | 0.094 | 0.300 [0.271, 0.329] | 72 / 783 / 605 | 0 | 148 |
| 0.8 | 1000 | 0.059 [0.046, 0.073] | 0.597 | 0.099 | 0.302 [0.274, 0.332] | 71 / 682 / 606 | 0 | 147 |
| 0.9 | 60 | 0.074 [0.060, 0.087] | 0.600 | 0.123 | 0.317 [0.287, 0.347] | 94 / 761 / 583 | 0 | 145 |
| 0.9 | 1000 | 0.077 [0.063, 0.092] | 0.601 | 0.129 | 0.319 [0.290, 0.349] | 91 / 647 / 586 | 0 | 144 |
| 0.95 | 60 | 0.084 [0.069, 0.098] | 0.612 | 0.137 | 0.330 [0.299, 0.360] | 105 / 754 / 572 | 1 | 140 |
| 0.95 | 1000 | 0.089 [0.073, 0.105] | 0.611 | 0.146 | 0.332 [0.302, 0.363] | 102 / 622 / 575 | 1 | 138 |
| 0.98 | 60 | 0.101 [0.086, 0.117] | 0.617 | 0.164 | 0.349 [0.318, 0.380] | 126 / 734 / 551 | 1 | 139 |
| **0.98** | **1000** | **0.108 [0.091, 0.126]** | **0.618** | **0.175** | **0.346 [0.313, 0.378]** | **119 / 565 / 558** | **1** | **136** |

These full-disk numbers are low: the model was trained on single-filament crops, so on whole images it produces
many false-positive components and merges neighbouring filaments. Larger inference resolutions are worse
(at thr 0.9, A 1000: PQ 0.077 at 1024, 0.053 at 1536, 0.032 at 2048). On training-style crops the same checkpoint
reaches Dice about 0.76 (0.7422 over all 677 validation crops), which confirms the code matches the training
setup. The released pipeline is a faithful, reproducible baseline, not a competitive full-disk solution.

The `best_val_dice = 0.7422` stored in the checkpoint is a **crop-level** Dice (crops centred on annotated
filaments); it is not a full-disk score and should not be compared with the leaderboard.

### Leaderboard

Public leaderboard (as of 2026-11-17): team **Kaiser**, rank **38 of 1,567**, score **0.55**; final submission
reference `58173877` (public 0.55 / private 0.55).

**Relationship to the leaderboard submission.** Submission `58173877` ("EXP-28") was obtained by a multi-stage
chain of experiments (EXP-9 ... EXP-28) in which earlier prediction files were combined and post-processed
instance by instance. That chain is **not part of this repository** and cannot be regenerated by the code here.
The code in this repository (model, training, inference, evaluation) is the self-contained component of our
work, and the predictions it generates (`predict.py`) have **not** been scored on the Kaggle leaderboard. We
therefore do not claim that this repository reproduces the leaderboard score.

## Technical report

`technical_report.tex` follows the organizers' `acmart` (sigconf) template. To compile it on Overleaf, add it to the
template project (which provides `preamble.tex` and `main.bib`) together with `refs_extra.bib` and the `figures/`
folder of this repository. The figures and the bootstrap intervals are regenerated by `make_report_figures.py`.

## Limitations

* The network is trained on single-filament crops and applied to full-disk images; instances are separated by
  connected-component labelling only, so touching filaments are merged and a filament broken by a bright gap
  is split. This is the main source of the gap between pixel Dice and Panoptic Quality.
* Training used one fixed split and one seed; no cross-validation or ensembling.
* Our PQ implementation and fragmentation counts are approximations of the official evaluation.

## License

MIT — see `LICENSE`. Contact: via Kaggle (`snnguynvnk19hl`).
