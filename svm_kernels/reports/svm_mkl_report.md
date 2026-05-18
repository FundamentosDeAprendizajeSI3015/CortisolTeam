# SVM with Improved Multiple Kernel Learning (MKL) — Crypto ML Project

**Generated:** 2026-05-17 18:18

**Source:** adapted from https://github.com/maospina1041/extremality_mkl

---

## 1. Improvements over the baseline
1. **Diverse kernel pool** — polynomial (d∈{2,3,4}) + RBF (4 γ values) + Laplacian (4 γ values), all on random feature subsets.
2. **All 4 metrics** — extremality ordering uses alignment, polarization, FSM, and complex_ratio simultaneously (vs. alignment + FSM only).
3. **C selection by CV** — SVM regularisation C chosen by 3-fold stratified cross-validation inside the training set for each kernel strategy.

---

## 2. Experimental Setup
- **Task:** Binary classification — crypto price Up (+1) / Down (-1).
- **Features:** 16 technical indicators.
- **Temporal split:** test = last 15% of dates.
- **Diverse pool:** 10 kernels, up to 8 features/kernel.
- **Poly degrees:** (2, 3, 4) | **Gamma scales:** (0.01, 0.1, 1.0, 10.0)
- **Extremality exponent (n):** 2
- **C grid:** [0.1, 1.0, 10.0, 100.0] (selected per-strategy by 3-fold CV)

---

## 3. Results

| Kernel Strategy | Accuracy | F1 | AUC | C used |
|-----------------|----------|----|-----|--------|
| Natural MKL | 0.5320 | 0.6509 | 0.5203 | 100.0 |
| Anti-Natural MKL ← best | 0.5405 | 0.6867 | 0.5277 | 100.0 |
| Uniform MKL | 0.5332 | 0.6457 | 0.5189 | 100.0 |
| RBF | 0.5179 | 0.6041 | 0.5128 | 100.0 |
| Polynomial (d=3) | 0.5174 | 0.6148 | 0.5066 | 100.0 |
| Linear | 0.5191 | 0.6216 | 0.5114 | 100.0 |

### Best strategy: **Anti-Natural MKL**
- Accuracy : 0.5405
- F1       : 0.6867
- AUC      : 0.5277
- C        : 100.0

---

## 4. Top 3 extremally-best kernels (Natural MKL)

| Rank | Kernel # | Weight | Alignment | Polarization | FSM | Complex Ratio |
|------|----------|--------|-----------|-------------|-----|---------------|
| 1 | #09 | 0.1148 | 0.0003 | -49.9965 | 53.1004 | 8.4790 |
| 2 | #08 | 0.1148 | 0.0002 | 74.9924 | 42.4261 | 6.4212 |
| 3 | #06 | 0.1148 | 0.0002 | -0.0370 | 44.5567 | 2000.0000 |

---

## 5. Interpretation
- Kernels with high **alignment** have geometry closely matching the class structure.
- High **FSM** indicates strong intra-class cohesion (small within-class spread, large between-class gap).
- The **anti-natural MKL** should consistently underperform, validating the extremality ordering.
- **C tuning** removes the shared-C confound between kernel strategies.

---

## 6. Figures
- `31_mkl_kernel_metrics.png` — Heat-map of all 4 metrics + natural weights.
- `32_mkl_performance_comparison.png` — Acc / F1 / AUC grouped bars.
- `33_mkl_weight_distribution.png` — Weight distribution and CDF.
- `34_mkl_metrics_vs_kernels.png` — AUC vs pool size sweep.
- `35_mkl_confusion_matrices.png` — 2×3 confusion matrices.
- `36_mkl_roc_curves.png` — ROC curves for all 6 strategies.

---

## 7. Related scripts (heart disease validation)
- `svm_kernels/base/heart_base.py`     — baseline MKL on heart dataset.
- `svm_kernels/improved/heart_improved.py` — improved MKL on heart dataset.
- `svm_kernels/comparison.py`           — baseline vs improved comparison.