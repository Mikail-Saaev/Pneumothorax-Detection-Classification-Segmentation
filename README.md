# 🫁 Pneumothorax Detection — Classification & Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Binary classification and pixel-wise segmentation of pneumothorax on chest X-rays (SIIM-ACR dataset). Two CNN architectures are compared, and a soft-voting ensemble combines their predictions.

---

## 📌 Project Overview

Pneumothorax (collapsed lung) is a life-threatening emergency that requires rapid diagnosis from chest X-rays. This project builds an end-to-end deep-learning pipeline that:

1. **Classifies** each radiograph as *healthy* or *pneumothorax* (binary classification).
2. **Segments** the affected lung region from the pixel-level RLE masks provided in the dataset.
3. **Compares** two fine-tuned architectures (ResNet-18 vs. DenseNet-121) and combines them into a soft-voting ensemble.

---

## 🏗️ Architecture

```
Input (DICOM → RGB) ─► ResNet-18 ──────────────────► BCEWithLogitsLoss
                    └─► DenseNet-121 (CheXNet head) ►   + ReduceLROnPlateau
                                                     └─► Ensemble (avg. probabilities)
```

| Model | Head | Params | Notes |
|---|---|---|---|
| **ResNet-18** | Dropout(0.5) → Linear(512→1) | ~11 M | Lightweight baseline |
| **DenseNet-121** | Dropout → 512 → 128 → 1 | ~8 M | CheXNet-inspired |
| **Ensemble** | Avg. probabilities | — | Reduces variance |

---

## 📁 Repository Structure

```
pneumothorax-detection/
├── README.md
├── requirements.txt
├── notebook/
│   └── pneumothorax_detection.ipynb   ← Main Colab notebook
└── src/                               ← Reusable Python package
    ├── __init__.py
    ├── config.py      ← Centralised hyperparameters (dataclass)
    ├── dataset.py     ← PyTorch Datasets (classification + segmentation)
    ├── models.py      ← ResNet-18 & DenseNet-121 definitions
    ├── train.py       ← Training loop with early stopping
    ├── evaluate.py    ← Metrics, plots (ROC, confusion matrix)
    └── utils.py       ← Seeding, device detection, DICOM/RLE helpers
```

---

## 🚀 Quick Start (Google Colab)

1. Open [`notebook/pneumothorax_detection.ipynb`](notebook/pneumothorax_detection.ipynb) in Google Colab.
2. Enable GPU: *Runtime → Change runtime type → T4 GPU*.
3. Run **Cell 1** (installs dependencies automatically).
4. Upload your data in **Cell 3**:
   - `trainSet-rle.csv` — CSV with `ImageId` and `EncodedPixels` columns.
   - A ZIP archive of DICOM files (`.dcm`).
5. Run all remaining cells end-to-end.

### Dataset

The SIIM-ACR Pneumothorax Segmentation dataset is available on Kaggle:  
🔗 https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

---

## ⚙️ Configuration

All hyperparameters live in `src/config.py` (a `dataclass`):

```python
cfg = Config(
    img_size   = 224,
    batch_size = 32,
    num_epochs = 15,
    learning_rate = 1e-4,
    patience   = 3,      # early-stopping patience
    dropout_rate = 0.5,
    threshold  = 0.5,    # decision threshold
)
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| **PPV** (Precision) | Fraction of positive predictions that are correct |
| **Recall** (Sensitivity) | Fraction of true positives detected |
| F1-Score | Harmonic mean of PPV and Recall |
| Specificity | True negative rate |
| NPV | Negative predictive value |
| **AUROC** | Area under the ROC curve |

> In a medical context, **Recall** is prioritised to minimise missed diagnoses (false negatives).

---

## 🐛 Bugs Fixed vs. Original Version

| # | Issue | Fix |
|---|---|---|
| 1 | `pretrained=True` deprecated in torchvision ≥ 0.13 | Replaced with `weights=ResNet18_Weights.IMAGENET1K_V1` |
| 2 | `model.state_dict().copy()` does **not** deep-copy tensors | Replaced with `copy.deepcopy(model.state_dict())` |
| 3 | `sys.exit(1)` crashes the Colab kernel | Replaced with `raise RuntimeError(...)` |
| 4 | `import glob` inside class methods | Moved to top-level imports |
| 5 | `metrics['f1-score']` KeyError in comparison chart | Unified key as `f1_score` throughout |
| 6 | Segmentation used a **random** mask placeholder | Replaced with actual RLE decoding via `decode_rle()` |
| 7 | Ensemble used `np.round()` (majority vote on 2 models always ties) | Replaced with averaged probabilities (soft vote) |
| 8 | All logic in one giant monolithic cell | Split into 12 focused, documented cells |

---

## 🔧 Dependencies

```
torch>=2.0.0    torchvision>=0.15.0    pydicom>=2.4.0
numpy           pandas                  scikit-learn
Pillow          opencv-python-headless  matplotlib
seaborn         tqdm                    albumentations
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📜 License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.
