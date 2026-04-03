#!/usr/bin/env python3
"""Metrics, evaluation helpers, and visualisation utilities."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


# ── Core metrics ──────────────────────────────────────────────────────────────

def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> Dict:
    """Compute a comprehensive set of binary-classification metrics.

    Args:
        predictions: Binary predictions (0 / 1).
        labels:      Ground-truth labels (0 / 1).
        probs:       Predicted probabilities – required for AUROC.

    Returns:
        Dict with keys: accuracy, recall, ppv, f1_score, specificity, npv, auroc,
        confusion_matrix (dict with tn, fp, fn, tp).
    """
    preds_np = predictions.numpy() if torch.is_tensor(predictions) else np.asarray(predictions)
    labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)

    recall = recall_score(labels_np, preds_np, zero_division=0)
    ppv = precision_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    accuracy = float(np.mean(preds_np == labels_np))

    cm = confusion_matrix(labels_np, preds_np)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class present in the batch
        tn = fp = fn = tp = 0

    specificity = tn / (tn + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    auroc = None
    if probs is not None:
        probs_np = probs.numpy() if torch.is_tensor(probs) else np.asarray(probs)
        fpr, tpr, _ = roc_curve(labels_np, probs_np)
        auroc = float(auc(fpr, tpr))

    return {
        "accuracy": accuracy,
        "recall": float(recall),
        "ppv": float(ppv),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "npv": float(npv),
        "auroc": auroc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_metrics(metrics: Dict, phase: str = "Validation") -> None:
    """Pretty-print a metrics dict."""
    print(f"\n  {phase} metrics")
    print(f"    Accuracy    : {metrics['accuracy']:.4f}")
    print(f"    PPV (Prec.) : {metrics['ppv']:.4f}")
    print(f"    Recall      : {metrics['recall']:.4f}")
    print(f"    F1-Score    : {metrics['f1_score']:.4f}")
    print(f"    Specificity : {metrics['specificity']:.4f}")
    print(f"    NPV         : {metrics['npv']:.4f}")
    if metrics.get("auroc") is not None:
        print(f"    AUROC       : {metrics['auroc']:.4f}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_training_curves(
    history1: Dict,
    history2: Dict,
    name1: str = "ResNet-18",
    name2: str = "DenseNet-121",
    save_path: str = "training_curves.png",
) -> None:
    """Side-by-side training curves for two models."""
    metrics = [
        ("train_loss", "val_loss", "Loss"),
        ("val_acc", None, "Val Accuracy"),
        ("val_ppv", None, "Val PPV"),
        ("val_recall", None, "Val Recall"),
        ("val_f1", None, "Val F1"),
        ("val_auroc", None, "Val AUROC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Training Curves — Classification Models", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    colors = {"m1_train": "#4C72B0", "m1_val": "#4C72B0",
              "m2_train": "#DD8452", "m2_val": "#DD8452"}

    for ax, (train_key, val_key, title) in zip(axes, metrics):
        data1 = history1.get(train_key, [])
        data1v = history1.get(val_key, []) if val_key else []
        data2 = history2.get(train_key, [])
        data2v = history2.get(val_key, []) if val_key else []

        # Filter None (AUROC may be None early in training)
        valid1 = [(i, v) for i, v in enumerate(data1) if v is not None]
        valid2 = [(i, v) for i, v in enumerate(data2) if v is not None]

        if valid1:
            x1, y1 = zip(*valid1)
            ax.plot(x1, y1, label=name1, color=colors["m1_train"], linewidth=2)
        if data1v:
            valid1v = [(i, v) for i, v in enumerate(data1v) if v is not None]
            if valid1v:
                x1v, y1v = zip(*valid1v)
                ax.plot(x1v, y1v, label=f"{name1} val", color=colors["m1_val"],
                        linewidth=2, linestyle="--")
        if valid2:
            x2, y2 = zip(*valid2)
            ax.plot(x2, y2, label=name2, color=colors["m2_train"], linewidth=2)
        if data2v:
            valid2v = [(i, v) for i, v in enumerate(data2v) if v is not None]
            if valid2v:
                x2v, y2v = zip(*valid2v)
                ax.plot(x2v, y2v, label=f"{name2} val", color=colors["m2_val"],
                        linewidth=2, linestyle="--")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


def plot_confusion_matrices(
    preds_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    names: List[str],
    save_path: str = "confusion_matrices.png",
) -> None:
    """Plot confusion matrices side by side."""
    fig, axes = plt.subplots(1, len(preds_list), figsize=(6 * len(preds_list), 5))
    if len(preds_list) == 1:
        axes = [axes]
    class_labels = ["Healthy", "Pneumothorax"]

    for ax, preds, labels, name in zip(axes, preds_list, labels_list, names):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={"size": 13},
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold", pad=12)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


def plot_roc_curves(
    probs_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    names: List[str],
    save_path: str = "roc_curves.png",
) -> None:
    """Overlay ROC curves for multiple models."""
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    plt.figure(figsize=(8, 7))

    for probs, labels, name, color in zip(probs_list, labels_list, names, palette):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


def build_comparison_table(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Build a formatted DataFrame comparing models across metrics.

    Args:
        metrics_dict: ``{model_name: metrics_dict}`` mapping.

    Returns:
        DataFrame with models as columns and metrics as rows.
    """
    metric_keys = [
        ("Accuracy", "accuracy"),
        ("PPV (Precision)", "ppv"),
        ("Recall (Sensitivity)", "recall"),
        ("F1-Score", "f1_score"),
        ("Specificity", "specificity"),
        ("NPV", "npv"),
        ("AUROC", "auroc"),
    ]
    data = {"Metric": [label for label, _ in metric_keys]}
    for model_name, m in metrics_dict.items():
        data[model_name] = [
            f"{m[key]:.4f}" if m.get(key) is not None else "N/A"
            for _, key in metric_keys
        ]
    return pd.DataFrame(data)
