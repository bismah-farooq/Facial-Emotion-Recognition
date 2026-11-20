# Facial Emotion Recognition (FER-2013)

CECS 456 Machine Learning — Project 2. Classify 48×48 grayscale face images into seven emotions
(angry, disgust, fear, happy, neutral, sad, surprise), comparing a CNN trained from scratch with a
fine-tuned ImageNet ResNet-18.

| Model | Owner | Code |
|---|---|---|
| **A — Custom CNN** (from scratch, ~4.8M params) | Bismah Farooq | [`src/models/custom_cnn.py`](src/models/custom_cnn.py) |
| **B — ResNet-18** (ImageNet transfer learning, ~11.2M params) | Jayden Woodall | [`src/models/resnet_fer.py`](src/models/resnet_fer.py) |

Shared code (data pipeline, training loop, evaluation, comparison) is in `src/`. Each person's model
file and training/evaluation runs are their individual contribution — see [Contributions](#contributions).

## Repository layout

```
src/
  data.py            loading, dedup, stratified train/val split, preprocessing & augmentation
  models/
    custom_cnn.py    Model A
    resnet_fer.py    Model B
  train.py           training loop (weighted CE, AdamW, cosine LR, early stopping on val macro-F1)
  evaluate.py        test metrics, confusion matrix, top confusions, correct/incorrect examples
  compare.py         learning curves, per-class F1 chart, markdown results table
scripts/
  analyze_data.py    class distribution, sample grid, pixel stats, duplicate check
notebooks/
  colab_train.ipynb  one-click GPU run on Google Colab
```

## 1. Setup

```bash
git clone https://github.com/bismah-farooq/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended (Colab's free T4 works; see the notebook). Apple-silicon `mps` and CPU are
supported but slow.

## 2. Get the data

Dataset: **FER-2013**, Kaggle — https://www.kaggle.com/datasets/msambare/fer2013 (folder version).

Either download the zip manually and unzip it so the layout is:

```
data/fer2013/train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
data/fer2013/test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
```

or use `kagglehub` (needs a Kaggle account / API token):

```bash
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('msambare/fer2013'); shutil.copytree(p, 'data/fer2013', dirs_exist_ok=True)"
```

## 3. Reproduce the results

```bash
# Dataset analysis (class imbalance, duplicates, samples)  -> outputs/data_analysis/
python scripts/analyze_data.py

# Model A - Custom CNN
python -m src.train    --model custom_cnn
python -m src.evaluate --run outputs/custom_cnn

# Model B - ResNet-18
python -m src.train    --model resnet18
python -m src.evaluate --run outputs/resnet18

# Side-by-side comparison  -> outputs/comparison/
python -m src.compare --runs outputs/custom_cnn outputs/resnet18
```

All runs use `--seed 42`. Useful options: `--epochs`, `--lr`, `--batch-size`, `--no-class-weights`
(imbalance ablation), `--no-pretrained` (ResNet without ImageNet weights); `src.evaluate --tta`
(horizontal-flip test-time augmentation). Run `python -m src.train -h` for the full list.

## Method summary

**Data split.** The official `test/` folder (7,178 images) is the held-out test set and is used only once, for
final evaluation. The official `train/` folder is split 90/10 (stratified) into train/validation.

**Duplicates.** FER-2013 contains exact pixel duplicates. Before splitting, `src/data.py` removes training
images whose pixels also appear in the test set (703) and repeated copies inside the training folder (1,064),
so no image — or augmented copy of one — appears in more than one split. Final sizes: **24,247 train /
2,695 val / 7,178 test**.

**Preprocessing.** Grayscale, normalise with the training-set mean/std (0.5049 / 0.2545).
**Augmentation (training only, on the fly):** horizontal flip, ±10° rotation, ±10% translation, 0.9–1.1 scale,
brightness/contrast jitter, random erasing (p = 0.25).

**Class imbalance.** "disgust" has 317 training images vs. 6,328 for "happy" (20× imbalance). We use
inverse-frequency class weights in the cross-entropy loss (plus label smoothing 0.1) and select checkpoints by
validation **macro-F1** rather than accuracy, so minority classes count equally.

**Model A — Custom CNN.** 4 VGG-style blocks (2× Conv3×3-BN-ReLU, MaxPool, Dropout) with 64/128/256/512
channels → global average pool → FC 256 → 7. AdamW, lr 1e-3, cosine schedule, up to 60 epochs.

**Model B — ResNet-18.** ImageNet weights. First conv replaced by a 1-channel 3×3 stride-1 conv initialised
from the centre of the pretrained kernel; initial max-pool removed so 48×48 inputs keep enough resolution;
new dropout + linear head. Backbone frozen for 3 epochs, then fine-tuned with 0.1× the head learning rate.
AdamW, lr 5e-4, cosine schedule, up to 40 epochs.

**Evaluation.** Accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrix
(counts + row-normalised), most frequent confusions, and grids of correct and high-confidence incorrect
predictions.

## Results

| Model | Run | Test accuracy | Macro-F1 |
|---|---|---|---|
| Custom CNN, inverse-frequency class weights | `outputs/custom_cnn` | 64.1% | 0.594 |
| Custom CNN, no class weights | `outputs/cnn_noweights` | 67.0% | 0.605 |
| **Custom CNN, sqrt class weights, 80 epochs** | `outputs/cnn_sqrtw` | **68.2%** | **0.653** |
| ResNet-18 (fine-tuned) | `outputs/resnet18` | 66.1% | 0.612 |

Full per-class table and top confusions: [`outputs/comparison/summary.md`](outputs/comparison/summary.md).
Reproduce the best Custom CNN run with
`python -m src.train --model custom_cnn --weight-power 0.5 --epochs 80 --run-name cnn_sqrtw`.

## Contributions

- **Bismah Farooq** — Model A (`src/models/custom_cnn.py`), its training and evaluation runs; dataset analysis
  and deduplication (`scripts/analyze_data.py`, `src/data.py`).
- **Jayden Woodall** — Model B (`src/models/resnet_fer.py`), its fine-tuning schedule, training and evaluation
  runs; model comparison (`src/compare.py`).

## References

- Goodfellow et al., "Challenges in Representation Learning: A report on three machine learning contests,"
  ICONIP 2013 (FER-2013 dataset). Kaggle mirror: https://www.kaggle.com/datasets/msambare/fer2013
- He et al., "Deep Residual Learning for Image Recognition," CVPR 2016 (ResNet). Pretrained weights from
  `torchvision` (`ResNet18_Weights.IMAGENET1K_V1`).
- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR 2015
  (VGG-style blocks used in Model A).
- PyTorch / torchvision, scikit-learn, matplotlib.
