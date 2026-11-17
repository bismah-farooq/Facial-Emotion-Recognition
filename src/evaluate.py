"""Evaluate a trained checkpoint on the held-out FER-2013 test set.

    python -m src.evaluate --run outputs/custom_cnn

Writes to the run folder:
    metrics.json                accuracy, macro-F1, per-class precision/recall/F1, top confusions
    predictions.csv             path, true label, predicted label, confidence
    confusion_matrix.png        counts and row-normalised (recall) matrices
    examples_correct.png        representative correct predictions (one row per class)
    examples_incorrect.png      representative errors (highest-confidence mistakes)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import CLASSES, FER_MEAN, FER_STD, load_splits
from .models import build_model
from .utils import get_device, save_json


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta:  # average with horizontally flipped input
            p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def plot_confusion(cm, path, title):
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mat, fmt, sub in [(axes[0], cm, "d", "counts"), (axes[1], cm_norm, ".2f", "row-normalised (recall)")]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} - {sub}")
        thresh = mat.max() / 2
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm, k=6):
    pairs = [(CLASSES[i], CLASSES[j], int(cm[i, j]), float(cm[i, j] / cm[i].sum()))
             for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j]
    pairs.sort(key=lambda t: -t[2])
    return [{"true": t, "pred": p, "count": c, "frac_of_true": round(f, 3)} for t, p, c, f in pairs[:k]]


def plot_examples(dataset, df, path, correct, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.6, len(CLASSES) * 1.9))
    for r, cls in enumerate(CLASSES):
        sub = df[(df.true == cls) & ((df.true == df.pred) == correct)]
        if not correct:  # most confident mistakes are the most informative
            sub = sub.sort_values("confidence", ascending=False).head(per_class * 3)
        pick = sub.sample(min(per_class, len(sub)), random_state=int(rng.integers(1e9))) if len(sub) else sub
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(pick):
                row = pick.iloc[c]
                img = dataset[int(row.idx)][0].squeeze().numpy() * FER_STD + FER_MEAN
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                color = "green" if correct else "red"
                ax.set_title(f"T:{row.true}\nP:{row.pred} ({row.confidence:.2f})", fontsize=7, color=color)
    fig.suptitle("Correct predictions" if correct else "Incorrect predictions (high-confidence errors)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run folder containing best.pt")
    p.add_argument("--data-dir", default="data/fer2013")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    a = p.parse_args()

    device = get_device()
    ckpt = torch.load(os.path.join(a.run, "best.pt"), map_location=device)
    model = build_model(ckpt["model"], len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, _, test_ds = load_splits(a.data_dir)
    loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, num_workers=2)
    probs = predict(model, loader, device, a.tta)
    y_true = np.array(test_ds.targets)
    y_pred = probs.argmax(1)

    df = pd.DataFrame({
        "idx": range(len(y_true)),
        "path": [os.path.relpath(s[0], a.data_dir) for s in test_ds.samples],
        "true": [CLASSES[i] for i in y_true],
        "pred": [CLASSES[i] for i in y_pred],
        "confidence": probs.max(1),
    })
    df.drop(columns="idx").to_csv(os.path.join(a.run, "predictions.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        "model": ckpt["model"],
        "best_epoch": ckpt["epoch"],
        "tta": a.tta,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "test_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in CLASSES},
        "top_confusions": top_confusions(cm),
        "confusion_matrix": cm.tolist(),
    }
    suffix = "_tta" if a.tta else ""
    save_json(metrics, os.path.join(a.run, f"metrics{suffix}.json"))

    title = {"custom_cnn": "Custom CNN", "resnet18": "ResNet-18"}.get(ckpt["model"], ckpt["model"])
    plot_confusion(cm, os.path.join(a.run, f"confusion_matrix{suffix}.png"), title)
    plot_examples(test_ds, df, os.path.join(a.run, "examples_correct.png"), correct=True)
    plot_examples(test_ds, df, os.path.join(a.run, "examples_incorrect.png"), correct=False)

    print(f"Test accuracy {metrics['test_accuracy']:.4f} | macro-F1 {metrics['test_macro_f1']:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3, zero_division=0))
    print("Most common confusions:")
    for t in metrics["top_confusions"]:
        print(f"  {t['true']:>8} -> {t['pred']:<8} {t['count']:4d}  ({t['frac_of_true']:.1%} of {t['true']})")


if __name__ == "__main__":
    main()
