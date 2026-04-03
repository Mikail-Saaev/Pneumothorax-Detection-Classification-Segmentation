#!/usr/bin/env python3
"""Training & validation loop with early stopping and model checkpointing."""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import calculate_metrics


# ── Single-epoch helpers ──────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
    model_name: str = "",
) -> Tuple[float, Dict]:
    """Run one full training epoch.

    Returns:
        Tuple of (mean loss, metrics dict).
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc=f"Train {model_name}", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach()
        preds = (probs > threshold).float()

        running_loss += loss.item() * images.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_preds, all_labels, all_probs)
    return epoch_loss, metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    model_name: str = "",
) -> Tuple[float, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Run one full validation / test epoch.

    Returns:
        Tuple of (mean loss, metrics dict, probabilities, predictions, labels).
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc=f"Val   {model_name}", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        running_loss += loss.item() * images.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_preds, all_labels, all_probs)
    return epoch_loss, metrics, all_probs.numpy(), all_preds.numpy(), all_labels.numpy()


# ── Full training run ─────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    model_name: str,
    epochs: int,
    patience: int,
    device: torch.device,
    threshold: float = 0.5,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Dict[str, List], np.ndarray, np.ndarray, np.ndarray]:
    """Train a classifier with early stopping based on validation F1-score.

    Checkpoints are saved whenever the validation F1-score improves.

    Args:
        model:            PyTorch model to train.
        train_loader:     DataLoader for training data.
        val_loader:       DataLoader for validation data.
        criterion:        Loss function.
        optimizer:        Optimiser instance.
        scheduler:        LR scheduler (``None`` disables scheduling).
        model_name:       Human-readable label used in logs and filenames.
        epochs:           Maximum number of epochs.
        patience:         Early-stopping patience.
        device:           Torch device.
        threshold:        Decision threshold for binary predictions.
        checkpoint_path:  File path to save the best model (``None`` = auto).

    Returns:
        (history, val_probs, val_preds, val_labels) from the best epoch.
    """
    if checkpoint_path is None:
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        checkpoint_path = f"best_{safe_name}.pth"

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "train_recall": [], "val_recall": [],
        "train_ppv": [], "val_ppv": [],
        "val_auroc": [],
    }

    best_val_f1 = -1.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_val_probs = best_val_preds = best_val_labels = None

    print(f"\n{'='*60}")
    print(f"  Training  {model_name}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        print(f"\n  Epoch {epoch:02d}/{epochs}")
        print(f"  {'─'*40}")

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, threshold, model_name
        )
        val_loss, val_m, val_probs, val_preds, val_labels = validate_one_epoch(
            model, val_loader, criterion, device, threshold, model_name
        )

        if scheduler is not None:
            scheduler.step(val_loss)
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ── Log history ───────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])
        history["train_f1"].append(train_m["f1_score"])
        history["val_f1"].append(val_m["f1_score"])
        history["train_recall"].append(train_m["recall"])
        history["val_recall"].append(val_m["recall"])
        history["train_ppv"].append(train_m["ppv"])
        history["val_ppv"].append(val_m["ppv"])
        history["val_auroc"].append(val_m.get("auroc"))

        # ── Console summary ───────────────────────────────────────────────────
        print(
            f"  Loss  — train: {train_loss:.4f}  val: {val_loss:.4f}\n"
            f"  Acc   — train: {train_m['accuracy']:.4f}  val: {val_m['accuracy']:.4f}\n"
            f"  F1    — train: {train_m['f1_score']:.4f}  val: {val_m['f1_score']:.4f}\n"
            f"  PPV   — train: {train_m['ppv']:.4f}        val: {val_m['ppv']:.4f}\n"
            f"  Recall— train: {train_m['recall']:.4f}    val: {val_m['recall']:.4f}"
        )
        if val_m.get("auroc") is not None:
            print(f"  AUROC —                               val: {val_m['auroc']:.4f}")

        # ── Checkpoint on best val F1 ─────────────────────────────────────────
        if val_m["f1_score"] > best_val_f1:
            best_val_f1 = val_m["f1_score"]
            best_model_weights = copy.deepcopy(model.state_dict())
            best_val_probs, best_val_preds, best_val_labels = val_probs, val_preds, val_labels
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_f1": best_val_f1,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"  ✔ New best model saved  (val F1 = {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after epoch {epoch}.")
                break

    # Restore best weights
    model.load_state_dict(best_model_weights)
    print(f"\n  ✔ Best weights restored  (val F1 = {best_val_f1:.4f})")
    return history, best_val_probs, best_val_preds, best_val_labels
