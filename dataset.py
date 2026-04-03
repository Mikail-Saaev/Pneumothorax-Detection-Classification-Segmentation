#!/usr/bin/env python3
"""PyTorch Datasets for pneumothorax classification and segmentation."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from .utils import decode_rle, find_dicom_file


# ── Classification dataset ────────────────────────────────────────────────────

class PneumothoraxClassificationDataset(Dataset):
    """Binary classification dataset (pneumothorax vs. healthy).

    Args:
        dataframe: DataFrame with columns 'ImageId' and 'has_pneumo'.
        root_dir:  Root directory containing DICOM files.
        transform: torchvision transform pipeline (default: resize + to-tensor).
        img_size:  Fallback image size used when the DICOM cannot be read.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform or transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )
        self.image_ids = self.dataframe["ImageId"].values
        self.labels = self.dataframe["has_pneumo"].values

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        label = self.labels[idx]
        image = self._load_image(img_id)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

    def _load_image(self, img_id: str) -> Image.Image:
        """Load a DICOM file and return a normalised RGB PIL Image."""
        try:
            path = find_dicom_file(self.root_dir, img_id)
            dcm = pydicom.dcmread(path)
            arr = dcm.pixel_array.astype(np.float32)
            # Min-max normalisation → [0, 255]
            lo, hi = arr.min(), arr.max()
            arr = (arr - lo) / (hi - lo + 1e-7) * 255.0
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        except Exception as exc:
            # Return a black image so training can continue despite missing files
            print(f"  [WARN] Could not load DICOM for '{img_id}': {exc}")
            return Image.new("RGB", (self.img_size, self.img_size), color=0)


# ── Segmentation dataset ──────────────────────────────────────────────────────

class PneumothoraxSegmentationDataset(Dataset):
    """Segmentation dataset returning (image_tensor, mask_tensor) pairs.

    The mask is decoded from the run-length encoding stored in the CSV and
    resized to *img_size × img_size*.

    Args:
        dataframe: DataFrame with columns 'ImageId' and 'EncodedPixels'.
        root_dir:  Root directory containing DICOM files.
        img_size:  Target spatial resolution.
        augment:   Whether to apply random horizontal flip augmentation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        img_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        img_id = row["ImageId"]
        rle_str = row["EncodedPixels"]

        # Load image
        cls_ds = PneumothoraxClassificationDataset(
            pd.DataFrame([{"ImageId": img_id, "has_pneumo": 1}]),
            self.root_dir,
            transform=None,
            img_size=self.img_size,
        )
        pil_image = cls_ds._load_image(img_id)
        orig_w, orig_h = pil_image.size

        # Decode RLE mask (original resolution)
        mask = decode_rle(rle_str, shape=(orig_h, orig_w))
        mask_pil = Image.fromarray(mask * 255).resize(
            (self.img_size, self.img_size), resample=Image.NEAREST
        )

        # Optional augmentation (shared flip)
        if self.augment and torch.rand(1).item() > 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

        image_tensor = self.img_transform(pil_image)
        mask_tensor = torch.from_numpy(
            (np.array(mask_pil) > 127).astype(np.float32)
        ).unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor
