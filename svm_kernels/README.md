# SVM with Multiple Kernel Learning (MKL)

**Phase 07 — CortisolTeam Crypto ML Project**

Adapted from [extremality_mkl](https://github.com/maospina1041/extremality_mkl) by maospina1041.

---

## Overview

Instead of committing to a single kernel (RBF, polynomial, or linear), this
module learns a **data-adaptive kernel** as a convex combination of many
*weak* polynomial kernels.  Each weak kernel is built from a random subset
of the technical indicator features (RSI, MACD, SMA, ATR, Bollinger Bands …)
and a random polynomial degree.

The **extremality ordering** algorithm then ranks these kernels across
multiple quality metrics simultaneously — without collapsing them into a
single scalar — and assigns higher weights to the kernels that are
collectively best for the binary price-direction classification task.

---

## Directory structure

```
svm_kernels/
├── src/
│   ├── kernel_metrics.py          # Alignment, Polarization, FSM, Complex Ratio
│   ├── weak_polynomial_kernel.py  # Random-feature-subset kernel generator
│   └── weight_combination.py      # Weighted kernel combination utilities
├── mkl/
│   ├── extremality_order.py       # Gram-Schmidt rotation + Pareto ranking
│   └── extremality_weights.py     # Full MKL weighting pipeline
├── reports/
│   ├── figures/                   # Generated plots (31–36)
│   └── svm_mkl_report.md          # Experiment summary
├── svm_mkl.py                     # Main pipeline script
└── README.md
```

---

## Kernel metrics

| Metric | Direction | Meaning |
|--------|-----------|---------|
| **Kernel Alignment** | ↑ higher is better | Frobenius inner product between K and the ideal kernel (perfect class separation). |
| **Kernel Polarization** | ↑ higher is better | Total margin between same-class and cross-class pairs in kernel space. |
| **FSM** | ↑ higher is better | Feature Space Measure — intra-class cohesion vs inter-class dispersion. |
| **Complex Ratio** | ↓ lower is better | Trace of K — penalises kernels that map data too far from the origin. |

---

## Kernel strategies compared

| Strategy | Description |
|----------|-------------|
| **Natural MKL** | Extremality-weighted sum; best kernels by Alignment + FSM get highest weight. |
| **Anti-Natural MKL** | Weights deliberately favour the worst-scoring kernels (ablation baseline). |
| **Uniform MKL** | Simple average of all weak kernels (no quality discrimination). |
| **RBF** | Standard radial basis function kernel (sklearn default). |
| **Polynomial (d=3)** | Homogeneous polynomial kernel of degree 3. |
| **Linear** | Linear dot-product kernel. |

---

## Usage

```bash
# From the repo root
python -m svm_kernels.svm_mkl

# Or directly
python svm_kernels/svm_mkl.py
```

> **Prerequisite:** `data/processed/features.parquet` must exist.
> Run `supervised/feature_engineering.py` first if needed.

---

## Outputs

| File | Description |
|------|-------------|
| `reports/figures/31_mkl_kernel_metrics.png` | Heat-map of metric values + natural MKL weights per weak kernel. |
| `reports/figures/32_mkl_performance_comparison.png` | Grouped bar chart: Accuracy / F1 / AUC for all 6 strategies. |
| `reports/figures/33_mkl_weight_distribution.png` | Natural vs anti-natural weight distribution and CDF. |
| `reports/figures/34_mkl_metrics_vs_kernels.png` | AUC vs number of weak kernels (sweep across 5–30 kernels). |
| `reports/figures/35_mkl_confusion_matrices.png` | 2×3 grid of confusion matrices. |
| `reports/figures/36_mkl_roc_curves.png` | ROC curves for all 6 strategies on a single plot. |
| `reports/svm_mkl_report.md` | Markdown report with results, top kernels, and interpretation. |

---

## Key parameters (`svm_mkl.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `NUM_KERNELS` | 30 | Total weak kernels generated. More → richer combination, slower. |
| `MAX_FEATURES` | 8 | Max features per weak kernel (out of 16 technical indicators). |
| `MAX_DEGREE` | 3 | Max polynomial degree per kernel. |
| `MKL_EXPONENT` | 2 | Weight sharpening. 1 = proportional; >1 = winner-takes-more. |
| `SVM_C` | 1.0 | SVM regularisation. Shared across all strategies for fair comparison. |

---

## Credit

Extremality ordering algorithm and kernel weighting strategy adapted from:

> **maospina1041** (2024). *extremality_mkl*.
> GitHub: https://github.com/maospina1041/extremality_mkl
