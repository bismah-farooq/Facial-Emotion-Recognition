"""Train a FER-2013 model.

Examples
    python -m src.train --model custom_cnn --epochs 60
    python -m src.train --model resnet18   --epochs 40 --freeze-epochs 3

Outputs (in outputs/<run-name>/):
    best.pt        best checkpoint by validation macro-F1
    history.json   per-epoch train/val loss, accuracy, macro-F1, lr
    config.json    all arguments, class counts and weights (for the report)
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .data import CLASSES, class_counts, class_weights, load_splits, make_loaders
from .models import build_model
from .utils import get_device, save_json, set_seed

DEFAULTS = {  # per-model defaults, overridable from the CLI
    "custom_cnn": dict(lr=1e-3, epochs=60, weight_decay=1e-4),
    "resnet18": dict(lr=5e-4, epochs=40, weight_decay=1e-4),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(DEFAULTS))
    p.add_argument("--data-dir", default="data/fer2013")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--run-name", default=None, help="defaults to the model name")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-class-weights", action="store_true", help="plain (unweighted) cross-entropy")
    p.add_argument("--patience", type=int, default=10, help="early stopping on val macro-F1")
    p.add_argument("--freeze-epochs", type=int, default=3, help="resnet18 only: epochs with frozen backbone")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1, help="resnet18 only: backbone lr = lr * mult")
    p.add_argument("--no-pretrained", action="store_true", help="resnet18 only: random init (ablation)")
    p.add_argument("--no-dedup", action="store_true", help="keep duplicate images (not recommended)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=None, help="debug: limit batches per epoch")
    a = p.parse_args()
    for k, v in DEFAULTS[a.model].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.run_name = a.run_name or a.model
    return a


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(tqdm(loader, leave=False, desc="train" if train else "val")):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.append(out.argmax(1).cpu())
            labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": total_loss / len(labels),
        "acc": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_ds, val_ds, test_ds = load_splits(args.data_dir, args.val_frac, args.seed, dedup=not args.no_dedup, verbose=True)
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, test_ds, args.batch_size, args.num_workers)
    counts = class_counts(train_ds)
    print(f"Device: {device} | train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")
    print("Train class counts:", dict(zip(CLASSES, counts)))

    weights = None if args.no_class_weights else class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=args.label_smoothing,
    )

    model = build_model(args.model, len(CLASSES), pretrained=not args.no_pretrained).to(device)
    if args.model == "resnet18":
        param_groups = [
            {"params": model.head_parameters(), "lr": args.lr},
            {"params": model.backbone_parameters(), "lr": args.lr * args.backbone_lr_mult},
        ]
        model.set_backbone_trainable(args.freeze_epochs == 0)
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    save_json(
        {**vars(args), "device": str(device), "n_params": n_params,
         "train_counts": dict(zip(CLASSES, counts)),
         "class_weights": None if weights is None else dict(zip(CLASSES, weights.tolist()))},
        os.path.join(run_dir, "config.json"),
    )
    print(f"{args.model}: {n_params/1e6:.2f}M parameters")

    history, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        if args.model == "resnet18" and epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            model.set_backbone_trainable(True)
            print("Unfroze ResNet backbone")
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, args.max_batches)
        va = run_epoch(model, val_loader, criterion, device, None, args.max_batches)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "time_s": round(time.time() - t0, 1),
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        save_json(history, os.path.join(run_dir, "history.json"))
        print(f"[{epoch:03d}] train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} F1 {va['macro_f1']:.3f} | {history[-1]['time_s']}s")

        if va["macro_f1"] > best_f1:
            best_f1, bad_epochs = va["macro_f1"], 0
            torch.save({"model": args.model, "state_dict": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best_f1, "args": vars(args)}, os.path.join(run_dir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                break
    print(f"Done. Best val macro-F1 {best_f1:.4f}. Checkpoint: {run_dir}/best.pt")


if __name__ == "__main__":
    main()
