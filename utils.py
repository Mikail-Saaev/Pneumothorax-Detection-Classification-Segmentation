#!/usr/bin/env python3
"""Utility helpers: seeding, device detection, RLE decoding, visualisation."""

import glob
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    """Print device summary."""
    print(f"  Device  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU     : {props.name}")
        print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB")


# ── DICOM helpers ─────────────────────────────────────────────────────────────

def find_dicom_file(root_dir: str, img_id: str) -> str:
    """Locate a DICOM file by image ID, searching recursively.

    Args:
        root_dir: Root directory to search in.
        img_id:   Image identifier (filename without extension).

    Returns:
        Absolute path to the first matching .dcm file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    # Fast exact-path check (avoids glob overhead for well-structured datasets)
    for ext in (".dcm", ".DCM"):
        candidate = os.path.join(root_dir, f"{img_id}{ext}")
        if os.path.exists(candidate):
            return candidate

    # Recursive search
    patterns = [
        os.path.join(root_dir, "**", f"{img_id}.dcm"),
        os.path.join(root_dir, "**", f"{img_id}.DCM"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    # Fallback: scan all DICOMs and match by stem
    all_dicoms = glob.glob(os.path.join(root_dir, "**", "*.dcm"), recursive=True)
    all_dicoms += glob.glob(os.path.join(root_dir, "**", "*.DCM"), recursive=True)
    for path in all_dicoms:
        if img_id in os.path.basename(path):
            return path

    raise FileNotFoundError(
        f"No DICOM file found for image_id='{img_id}' under '{root_dir}'"
    )


# ── RLE decoding ──────────────────────────────────────────────────────────────

def decode_rle(rle_str: str, shape: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Decode a run-length encoded string into a binary mask.

    Args:
        rle_str: RLE string from the dataset CSV. '-1' means no mask.
        shape:   (height, width) of the original image.

    Returns:
        Binary mask as uint8 ndarray of the given shape.
    """
    if not rle_str or str(rle_str).strip() in ("-1", "nan"):
        return np.zeros(shape, dtype=np.uint8)
    try:
        tokens = str(rle_str).split()
        starts = np.asarray(tokens[0::2], dtype=int) - 1  # 1-indexed → 0-indexed
        lengths = np.asarray(tokens[1::2], dtype=int)
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, length in zip(starts, lengths):
            mask[start : start + length] = 1
        return mask.reshape(shape).T  # column-major order used in the dataset
    except Exception:
        return np.zeros(shape, dtype=np.uint8)
