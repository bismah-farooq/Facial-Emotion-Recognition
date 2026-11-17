"""Compare evaluated runs side by side (for the Results section of the report).

    python -m src.compare --runs outputs/custom_cnn outputs/resnet18

Writes to outputs/comparison/:
    learning_curves.png   loss / accuracy / macro-F1 vs. epoch for every run
    per_class_f1.png      grouped bar chart of test F1 per emotion
    summary.md            markdown table ready to paste into the report
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES
from .utils import load_json

NAMES = {"custom_cnn": "Custom CNN", "resnet18": "ResNet-18 (fine-tuned)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="outputs/comparison")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = []
    for r in a.runs:
        m = load_json(os.path.join(r, "metrics.json"))
        h = load_json(os.path.join(r, "history.json"))
        c = load_json(os.path.join(r, "config.json"))
        label = NAMES.get(m["model"], m["model"])
        if os.path.basename(os.path.normpath(r)) not in (m["model"],):
            label += f" [{os.path.basename(os.path.normpath(r))}]"
        runs.append((label, m, h, c))

    # Learning curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for label, _, h, _ in runs:
        ep = [e["epoch"] for e in h]
        for ax, key in zip(axes, ("loss", "acc", "macro_f1")):
            line, = ax.plot(ep, [e[f"train_{key}"] for e in h], "--", alpha=0.7, label=f"{label} train")
            ax.plot(ep, [e[f"val_{key}"] for e in h], "-", color=line.get_color(), label=f"{label} val")
    for ax, t in zip(axes, ("Loss", "Accuracy", "Macro-F1")):
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "learning_curves.png"), dpi=150)
    plt.close(fig)

    # Per-class F1
    x = np.arange(len(CLASSES))
    w = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (label, m, _, _) in enumerate(runs):
        ax.bar(x + i * w - 0.4 + w / 2, [m["per_class"][c]["f1-score"] for c in CLASSES], w, label=label)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class F1 on FER-2013 test set")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "per_class_f1.png"), dpi=150)
    plt.close(fig)

    # Markdown summary
    lines = ["| Model | Params | Best epoch | Test acc | Macro-F1 | Weighted-F1 |",
             "|---|---|---|---|---|---|"]
    for label, m, _, c in runs:
        lines.append(f"| {label} | {c['n_params']/1e6:.2f}M | {m['best_epoch']} | "
                     f"{m['test_accuracy']:.4f} | {m['test_macro_f1']:.4f} | {m['test_weighted_f1']:.4f} |")
    lines += ["", "| Class | " + " | ".join(l for l, *_ in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for cls in CLASSES:
        lines.append(f"| {cls} | " + " | ".join(f"{m['per_class'][cls]['f1-score']:.3f}" for _, m, _, _ in runs) + " |")
    lines += ["", "Top confusions (true -> predicted):"]
    for label, m, _, _ in runs:
        tops = ", ".join(f"{t['true']}->{t['pred']} ({t['count']})" for t in m["top_confusions"][:4])
        lines.append(f"- {label}: {tops}")
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
