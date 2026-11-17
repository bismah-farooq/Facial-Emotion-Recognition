"""Dataset analysis for the report's "Dataset" section.

    python scripts/analyze_data.py --data-dir data/fer2013

Writes to outputs/data_analysis/:
    class_distribution.png   train/val/test counts per emotion (class imbalance)
    samples.png              random examples from each class
    stats.json               counts, imbalance ratio, pixel mean/std,
                             exact-duplicate images within and across splits
"""
import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import datasets  # noqa: E402
from src.data import CLASSES, load_splits, class_counts, dedup_train_indices  # noqa: E402
from src.utils import save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/fer2013")
    p.add_argument("--out", default="outputs/data_analysis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw_train = datasets.ImageFolder(os.path.join(a.data_dir, "train"))
    raw_test = datasets.ImageFolder(os.path.join(a.data_dir, "test"))
    _, removed = dedup_train_indices(raw_train, raw_test, a.data_dir)
    train_ds, val_ds, test_ds = load_splits(a.data_dir)  # deduplicated splits used for training
    counts = {"train": class_counts(train_ds), "val": class_counts(val_ds), "test": class_counts(test_ds)}

    # Class distribution
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (split, c) in enumerate(counts.items()):
        bars = ax.bar(x + (i - 1) * 0.27, c, 0.27, label=split)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Images")
    ax.set_title("FER-2013 class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # Sample grid (raw images)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(CLASSES), 8, figsize=(10, 9))
    by_class = defaultdict(list)
    for path, lbl in test_ds.samples:
        by_class[lbl].append(path)
    for r, cls in enumerate(CLASSES):
        for c, path in enumerate(rng.choice(by_class[r], 8, replace=False)):
            axes[r, c].imshow(Image.open(path).convert("L"), cmap="gray")
            axes[r, c].axis("off")
        axes[r, 0].set_title(cls, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "samples.png"), dpi=150)
    plt.close(fig)

    # Pixel stats on training split + duplicate detection on all images
    folder = train_ds.folder
    tr_paths = [folder.samples[i][0] for i in train_ds.indices]
    va_paths = [folder.samples[i][0] for i in val_ds.indices]
    te_paths = [s[0] for s in test_ds.samples]
    s = ss = n = 0
    hashes = {}
    for split, paths in (("train", tr_paths), ("val", va_paths), ("test", te_paths)):
        for pth in paths:
            arr = np.asarray(Image.open(pth).convert("L"), dtype=np.float64) / 255.0
            if split == "train":
                s += arr.sum(); ss += (arr ** 2).sum(); n += arr.size
            hashes.setdefault(hashlib.md5(arr.tobytes()).hexdigest(), []).append(split)
    mean = s / n
    std = (ss / n - mean ** 2) ** 0.5
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    cross = Counter(tuple(sorted(set(v))) for v in dup_groups if len(set(v)) > 1)

    tr = counts["train"]
    stats = {
        "counts": {k: dict(zip(CLASSES, v)) for k, v in counts.items()},
        "totals": {k: sum(v) for k, v in counts.items()},
        "imbalance_ratio_max_over_min": max(tr) / min(tr),
        "pixel_mean": mean,
        "pixel_std": std,
        "raw_train_images": len(raw_train),
        "dedup_removed": removed,
        "after_dedup_duplicate_groups": len(dup_groups),
        "after_dedup_groups_spanning_splits": {"+".join(k): v for k, v in cross.items()},
    }
    save_json(stats, os.path.join(a.out, "stats.json"))
    print(f"Train/val/test: {stats['totals']}")
    print(f"Imbalance (largest/smallest class): {stats['imbalance_ratio_max_over_min']:.1f}x")
    print(f"Pixel mean {mean:.4f}, std {std:.4f}")
    print(f"Dedup removed from train folder: {removed}")
    print(f"After dedup: {len(dup_groups)} duplicate groups (within test only); "
          f"spanning splits: {stats['after_dedup_groups_spanning_splits'] or 'none'}")


if __name__ == "__main__":
    main()
