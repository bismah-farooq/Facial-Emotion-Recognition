"""FER-2013 data loading, splitting, preprocessing and augmentation.

Expected layout (Kaggle "msambare/fer2013"):
    data/fer2013/train/<emotion>/*.jpg
    data/fer2013/test/<emotion>/*.jpg

The official train folder is deduplicated (see dedup_train_indices) and then
split (stratified) into train/validation.
The official test folder is held out and only used for final evaluation.
Augmentation is applied on the fly to training batches only, so augmented
copies never leak into the validation or test splits.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 48

# Mean/std of deduplicated FER-2013 training pixels (grayscale, scaled to [0, 1]).
# Recompute with scripts/dataset_stats.py if you change the data.
FER_MEAN, FER_STD = 0.5049, 0.2545


def build_transforms(train: bool, img_size: int = IMG_SIZE):
    """Return the preprocessing pipeline.

    Training: random horizontal flip, small rotation/shift/scale, mild
    brightness/contrast jitter, and random erasing (simulates occlusion
    by hands/glasses). Eval: just grayscale -> tensor -> normalize.
    """
    resize = [transforms.Resize((img_size, img_size))] if img_size != IMG_SIZE else []
    base = [transforms.Grayscale(num_output_channels=1), *resize]
    norm = [transforms.ToTensor(), transforms.Normalize([FER_MEAN], [FER_STD])]
    if not train:
        return transforms.Compose(base + norm)
    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    return transforms.Compose(
        base + aug + norm + [transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))]
    )


class _TransformedSubset(Dataset):
    """Subset view of an ImageFolder that applies its own transform."""

    def __init__(self, folder: datasets.ImageFolder, indices, transform):
        self.folder, self.indices, self.transform = folder, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.folder.samples[self.indices[i]]
        img = self.folder.loader(path)
        return self.transform(img), label

    @property
    def targets(self):
        return [self.folder.targets[j] for j in self.indices]


def _pixel_hash(path: str) -> str:
    return hashlib.md5(np.asarray(Image.open(path).convert("L")).tobytes()).hexdigest()


def _hashes(folder: datasets.ImageFolder, cache_path: str) -> list[str]:
    """MD5 of decoded pixels for every image (cached to disk: ~20 s to compute)."""
    rel = [os.path.relpath(p, os.path.dirname(cache_path)) for p, _ in folder.samples]
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if all(r in cache for r in rel):
            return [cache[r] for r in rel]
    hashes = [_pixel_hash(p) for p, _ in folder.samples]
    try:
        with open(cache_path, "w") as f:
            json.dump(dict(zip(rel, hashes)), f)
    except OSError:
        pass
    return hashes


def dedup_train_indices(train: datasets.ImageFolder, test: datasets.ImageFolder, data_dir: str):
    """Indices of training images to keep so that no image appears in two splits.

    FER-2013 contains exact pixel duplicates both inside the training folder and
    between the training and test folders. We (1) drop any training image whose
    pixels also appear in the test set and (2) keep only the first copy of any
    image repeated inside the training folder, so the train/val split below
    cannot place copies of one image on both sides.
    """
    tr_h = _hashes(train, os.path.join(data_dir, ".hash_cache_train.json"))
    te_h = set(_hashes(test, os.path.join(data_dir, ".hash_cache_test.json")))
    seen, keep, stats = set(), [], Counter()
    for i, h in enumerate(tr_h):
        if h in te_h:
            stats["in_test"] += 1
        elif h in seen:
            stats["dup_in_train"] += 1
        else:
            seen.add(h)
            keep.append(i)
    return np.array(keep), dict(stats)


def load_splits(data_dir: str, val_frac: float = 0.1, seed: int = 42, img_size: int = IMG_SIZE,
                dedup: bool = True, verbose: bool = False):
    """Return (train_ds, val_ds, test_ds).

    Validation is a stratified slice of the (deduplicated) official train folder;
    the official test folder is left untouched.
    """
    train_root = os.path.join(data_dir, "train")
    test_root = os.path.join(data_dir, "test")
    for p in (train_root, test_root):
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing {p}. See README 'Get the data'.")

    full = datasets.ImageFolder(train_root)
    assert full.classes == CLASSES, f"Unexpected class folders: {full.classes}"
    if dedup:
        idx, removed = dedup_train_indices(full, datasets.ImageFolder(test_root), data_dir)
        if verbose:
            print(f"Dedup: removed {removed.get('in_test', 0)} train images also in test, "
                  f"{removed.get('dup_in_train', 0)} repeated train images")
    else:
        idx = np.arange(len(full))
    from sklearn.model_selection import train_test_split  # lazy: keeps the webcam demo light

    targets = np.array(full.targets)[idx]
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_frac, stratify=targets, random_state=seed
    )
    train_ds = _TransformedSubset(full, tr_idx, build_transforms(True, img_size))
    val_ds = _TransformedSubset(full, va_idx, build_transforms(False, img_size))
    test_ds = datasets.ImageFolder(test_root, transform=build_transforms(False, img_size))
    return train_ds, val_ds, test_ds


def class_counts(ds) -> list[int]:
    c = Counter(ds.targets)
    return [c.get(i, 0) for i in range(len(CLASSES))]


def class_weights(ds) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1 (for weighted cross-entropy)."""
    counts = torch.tensor(class_counts(ds), dtype=torch.float)
    w = counts.sum() / (len(counts) * counts)
    return w / w.mean()


def make_loaders(train_ds, val_ds, test_ds, batch_size=64, num_workers=2):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
