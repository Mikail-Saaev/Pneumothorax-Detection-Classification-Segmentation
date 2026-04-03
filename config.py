#!/usr/bin/env python3
"""Centralised hyperparameters & paths for the Pneumothorax Detection project."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: str = "/content/pneumothorax_data"

    # ── Image preprocessing ───────────────────────────────────────────────────
    img_size: int = 224
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)  # ImageNet statistics
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 2
    num_epochs_classification: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.5
    patience: int = 3          # early-stopping patience (epochs)
    threshold: float = 0.5     # decision threshold for binary classification

    # ── Data splits ──────────────────────────────────────────────────────────
    val_size: float = 0.15
    test_size: float = 0.15    # fraction of the total dataset
