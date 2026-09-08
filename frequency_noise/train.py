"""
TrueSight — frequency model training.

Reuses spatial/losses.py (FocalLoss, compute_class_weights) and
spatial/metrics.py (compute_binary_metrics) rather than duplicating them
-- these are generic binary-classification utilities, not spatial-stream-
specific, so every stream should share one implementation.

Usage:
    python -m frequency_noise.train --epochs 3 --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from frequency_noise.dataset import FrequencyDataset
from frequency_noise.frequency_model import build_model
from spatial.losses import FocalLoss, compute_class_weights
from spatial.metrics import compute_binary_metrics

CHECKPOINT_DIR = Path("models/frequency")


def make_balanced_sampler(dataset: FrequencyDataset) -> WeightedRandomSampler:
    labels = [dataset.base.rows[i]["label"] for i in range(len(dataset))]
    int_labels = [0 if l == "real" else 1 for l in labels]
    class_weights = compute_class_weights(int_labels)
    sample_weights = [class_weights[l] for l in int_labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu") -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, labels, _meta in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_binary_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss
    return metrics


def train(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,          # higher LR than spatial: small from-scratch model, no pretrained warm-start
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training frequency stream on device: {device}")

    train_ds = FrequencyDataset(split="train")
    val_ds = FrequencyDataset(split="val")

    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model().to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    history = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.3f} "
            f"val_auc={val_metrics['roc_auc']}"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint_path = checkpoint_dir / "frequency_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint to {checkpoint_path} (val_f1={best_val_f1:.3f})")

    return {"history": history, "best_val_f1": best_val_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
