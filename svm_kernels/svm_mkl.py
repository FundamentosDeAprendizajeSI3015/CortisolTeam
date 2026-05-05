"""
[07] SVM with Multiple Kernel Learning (MKL) — Crypto ML Project
================================================================
Adapted from https://github.com/maospina1041/extremality_mkl

This module applies the Extremality-MKL framework to the cryptocurrency
price-direction classification task (target: will the closing price be
higher tomorrow than today?).

Improvements over the original baseline (inspired by heart_improved.py)
------------------------------------------------------------------------
  1. Diverse kernel pool: polynomial + RBF (multi-gamma) + Laplacian (multi-gamma).
     Instead of only random-subset polynomial kernels, the pool now mixes three
     kernel families so the extremality ordering can select complementary
     representations of the technical indicator feature space.

  2. All 4 metrics for extremality weighting: alignment, polarization, FSM,
     and complex_ratio are all used simultaneously to rank weak kernels,
     rather than only alignment + FSM.  This gives the Pareto dominance ordering
     a richer multi-criteria view of kernel quality.

  3. C selection by internal CV: the SVM regularisation parameter C is chosen
     by a fast 3-fold stratified cross-validation inside the training set for
     each kernel strategy, rather than using a fixed value.

Kernel strategies compared
--------------------------
  - Natural MKL:      extremality-weighted; best kernels by all 4 metrics.
  - Anti-Natural MKL: ablation — deliberately worst-scoring kernels.
  - Uniform MKL:      simple average of the diverse pool (no weighting).
  - RBF:              standard global RBF (gamma = scale from sklearn).
  - Polynomial (d=3): standard homogeneous polynomial kernel.
  - Linear:           dot-product kernel.

Outputs
-------
    svm_kernels/reports/figures/31_mkl_kernel_metrics.png
    svm_kernels/reports/figures/32_mkl_performance_comparison.png
    svm_kernels/reports/figures/33_mkl_weight_distribution.png
    svm_kernels/reports/figures/34_mkl_metrics_vs_kernels.png
    svm_kernels/reports/figures/35_mkl_confusion_matrices.png
    svm_kernels/reports/figures/36_mkl_roc_curves.png
    svm_kernels/reports/svm_mkl_report.md

Usage
-----
    python -m svm_kernels.svm_mkl
    # or from repo root:
    python svm_kernels/svm_mkl.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

from svm_kernels.src.kernel_metrics import (
    kernel_alignment,
    kernel_polarization,
    fsm,
    complex_ratio,
)
from svm_kernels.src.diverse_kernel_pool import create_diverse_kernels
from svm_kernels.src.weight_combination import combine_kernels
from svm_kernels.mkl.extremality_weights import (
    compute_extremality_weights,
    compute_kernel_metrics,
    ALL_METRICS,
)

# ── Paths ───────────────────────────────────────────────────────────────────

DATA_PATH   = ROOT / 'data' / 'processed' / 'features.parquet'
REPORTS_DIR = ROOT / 'svm_kernels' / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment constants ────────────────────────────────────────────────────

# Same feature set as the supervised pipeline for fair comparability
BASE_FEATURES = [
    'sma_7', 'sma_14', 'sma_30',
    'ema_14', 'rsi_14',
    'bb_high', 'bb_low', 'bb_width',
    'macd', 'macd_signal',
    'atr_14', 'stoch_k', 'stoch_d', 'williams_r', 'obv',
    'Return',
]

# Diverse pool parameters (improvement 1)
NUM_KERNELS   = 30    # total weak kernels in the diverse pool
MAX_FEATURES  = 8     # max features sampled per kernel (out of 16)
POLY_DEGREES  = (2, 3, 4)
GAMMA_SCALES  = (0.01, 0.1, 1.0, 10.0)   # multiplied by 1/n_features internally

# Extremality weighting (improvement 2 — all 4 metrics)
MKL_EXPONENT  = 2     # weight sharpening; n>1 = winner-takes-more

# C selection (improvement 3)
C_GRID        = [0.1, 1.0, 10.0, 100.0]  # candidates for cross-validation
CV_FOLDS      = 3                          # internal CV folds for C selection

RANDOM_STATE  = 42
TEST_FRACTION = 0.15

# Sweep parameters for the ablation plot (Fig 34)
KERNEL_SWEEP  = [5, 10, 15, 20, 25, 30]
SWEEP_ITERS   = 3


# ── Data loading ────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features.parquet and return a temporally-split train/test set.

    The split is strictly temporal (no shuffle) to avoid look-ahead bias in
    the time-series of crypto prices.  Labels are converted from {0, 1} to
    {-1, +1} as required by the MKL kernel metric formulas.

    Returns:
        (X_train, X_test, y_train, y_test) — all as MinMaxScaler-normalised
        numpy arrays with labels in {-1, +1}.

    Raises:
        FileNotFoundError: If features.parquet has not been generated yet.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'{DATA_PATH} not found. '
            'Run supervised/feature_engineering.py first.'
        )

    df = pd.read_parquet(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    df = df.dropna(subset=BASE_FEATURES + ['target'])

    # Temporal split — cut on unique date axis to avoid per-symbol leakage
    dates  = df['Date'].sort_values().unique()
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]
    train  = df[df['Date'] < cutoff]
    test   = df[df['Date'] >= cutoff]

    print(f'  Train: {len(train):>6} rows  '
          f'({train["Date"].min().date()} → {train["Date"].max().date()})')
    print(f'  Test : {len(test):>6} rows  '
          f'({test["Date"].min().date()} → {test["Date"].max().date()})')

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(train[BASE_FEATURES].values)
    X_test  = scaler.transform(test[BASE_FEATURES].values)

    # Convert binary labels {0,1} → {-1,+1}  (required by FSM, alignment, etc.)
    y_train = np.where(train['target'].values == 0, -1, 1)
    y_test  = np.where(test['target'].values  == 0, -1, 1)

    print(f'  Train label balance:  '
          f'-1: {(y_train == -1).sum()}  +1: {(y_train == 1).sum()}')
    return X_train, X_test, y_train, y_test


# ── C selection by internal CV (improvement 3) ─────────────────────────────

def select_c(
    K_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: list[float] = C_GRID,
    n_splits: int = CV_FOLDS,
) -> float:
    """Choose the SVM regularisation C by stratified k-fold CV.

    Adapted from heart_improved.py.  Operates on the precomputed kernel
    sub-matrix for each fold so that the internal CV uses the correct
    train-only rows/columns.

    Args:
        K_train: Full training Gram matrix (n_train, n_train).
        y_train: Training labels in {-1, +1}.
        c_grid:  Candidate C values to try.
        n_splits: Number of CV folds.

    Returns:
        Best C value from the grid.
    """
    # Convert labels to {0,1} for StratifiedKFold (needs non-negative classes)
    y_01 = (y_train == 1).astype(int)
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    best_c, best_score = c_grid[0], -1.0
    for c in c_grid:
        fold_scores = []
        for tr_idx, va_idx in skf.split(K_train, y_01):
            # Extract sub-matrices for this fold
            Ktr = K_train[np.ix_(tr_idx, tr_idx)]
            Kva = K_train[np.ix_(va_idx, tr_idx)]
            clf = SVC(kernel='precomputed', C=c)
            clf.fit(Ktr, y_train[tr_idx])
            fold_scores.append(
                accuracy_score(y_train[va_idx], clf.predict(Kva))
            )
        mean = float(np.mean(fold_scores))
        if mean > best_score:
            best_score, best_c = mean, c
    return best_c


# ── SVM evaluation ──────────────────────────────────────────────────────────

def evaluate_svm(
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    tune_c: bool = True,
) -> dict:
    """Train an SVM with a precomputed kernel and return classification metrics.

    Args:
        K_train:  Gram matrix for training, shape (n_train, n_train).
        K_test:   Gram matrix for testing,  shape (n_test,  n_train).
        y_train:  Training labels in {-1, +1}.
        y_test:   Test labels    in {-1, +1}.
        tune_c:   If True, select C by internal CV (improvement 3).
                  If False, use C=1.0 (original baseline behaviour).

    Returns:
        Dictionary with keys: accuracy, f1, auc, C_used, y_pred, y_score.
    """
    C = select_c(K_train, y_train) if tune_c else 1.0

    clf = SVC(kernel='precomputed', C=C, probability=True, random_state=RANDOM_STATE)
    clf.fit(K_train, y_train)
    y_pred  = clf.predict(K_test)
    y_score = clf.predict_proba(K_test)[:, 1]

    y_test_01 = (y_test  == 1).astype(int)
    y_pred_01 = (y_pred  == 1).astype(int)

    return {
        'accuracy': float(accuracy_score(y_test_01, y_pred_01)),
        'f1':       float(f1_score(y_test_01, y_pred_01, zero_division=0)),
        'auc':      float(roc_auc_score(y_test_01, y_score)),
        'C_used':   C,
        'y_pred':   y_pred,
        'y_score':  y_score,
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_mkl_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_kernels: int = NUM_KERNELS,
    random_state: int = RANDOM_STATE,
    tune_c: bool = True,
) -> dict:
    """Execute the full improved MKL pipeline.

    Steps:
        1. Generate a diverse pool of weak kernels (poly + RBF + Laplacian).
        2. Rank kernels using all 4 quality metrics via extremality ordering.
        3. Build Natural, Anti-Natural, and Uniform combined Gram matrices.
        4. Add three standard-kernel baselines (RBF, Polynomial, Linear).
        5. Select C per strategy by internal CV, then train and evaluate SVM.

    Args:
        X_train, X_test: MinMaxScaler-normalised feature arrays.
        y_train, y_test: Labels in {-1, +1}.
        num_kernels:     Diverse pool size.
        random_state:    RNG seed.
        tune_c:          Whether to tune C by CV (True) or use C=1.0 (False).

    Returns:
        Dictionary with keys:
            'results'  — {kernel_name: metrics_dict}
            'ew'       — ExtremalityWeights (weights + measures)
            'KL_train' — diverse kernel stack (num_kernels, n_train, n_train)
            'KL_test'  — diverse kernel stack (num_kernels, n_test, n_train)
    """
    print(f'\n  Generating diverse pool of {num_kernels} weak kernels '
          f'(poly + RBF + Laplacian) ...')
    KL_train, KL_test = create_diverse_kernels(
        X_train, X_test,
        num_kernels=num_kernels,
        max_features=MAX_FEATURES,
        poly_degrees=POLY_DEGREES,
        gamma_scales=GAMMA_SCALES,
        random_state=random_state,
    )

    print('  Computing extremality weights (all 4 metrics) ...')
    ew = compute_extremality_weights(KL_train, y_train, exponent=MKL_EXPONENT)

    # ── Build combined Gram matrices ────────────────────────────────────────

    # Natural MKL — highest weight for extremally-best kernels
    K_nat_tr  = combine_kernels(KL_train, ew.w_natural)
    K_nat_te  = combine_kernels(KL_test,  ew.w_natural)

    # Anti-Natural MKL — adversarial ablation (worst kernels)
    K_anti_tr = combine_kernels(KL_train, ew.w_anti_natural)
    K_anti_te = combine_kernels(KL_test,  ew.w_anti_natural)

    # Uniform MKL — equal weights (no quality discrimination)
    unif = np.ones(num_kernels) / num_kernels
    K_uni_tr  = combine_kernels(KL_train, unif)
    K_uni_te  = combine_kernels(KL_test,  unif)

    # Standard kernels on the raw (normalised) feature vectors
    K_rbf_tr  = rbf_kernel(X_train)
    K_rbf_te  = rbf_kernel(X_test, X_train)
    K_poly_tr = polynomial_kernel(X_train, degree=3, gamma=1.0, coef0=0)
    K_poly_te = polynomial_kernel(X_test, X_train, degree=3, gamma=1.0, coef0=0)
    K_lin_tr  = linear_kernel(X_train)
    K_lin_te  = linear_kernel(X_test, X_train)

    # ── Evaluate all strategies ─────────────────────────────────────────────
    print(f'  Evaluating SVMs (C tuning: {tune_c}) ...')
    variants = {
        'Natural MKL':      (K_nat_tr,  K_nat_te),
        'Anti-Natural MKL': (K_anti_tr, K_anti_te),
        'Uniform MKL':      (K_uni_tr,  K_uni_te),
        'RBF':              (K_rbf_tr,  K_rbf_te),
        'Polynomial (d=3)': (K_poly_tr, K_poly_te),
        'Linear':           (K_lin_tr,  K_lin_te),
    }

    results = {}
    for name, (Ktr, Kte) in variants.items():
        m = evaluate_svm(Ktr, Kte, y_train, y_test, tune_c=tune_c)
        results[name] = m
        print(f'    {name:<22}  Acc={m["accuracy"]:.3f}  '
              f'F1={m["f1"]:.3f}  AUC={m["auc"]:.3f}  C={m["C_used"]}')

    return {'results': results, 'ew': ew, 'KL_train': KL_train, 'KL_test': KL_test}


# ── Plotting ─────────────────────────────────────────────────────────────────

PALETTE = {
    'Natural MKL':      'steelblue',
    'Anti-Natural MKL': 'tomato',
    'Uniform MKL':      'mediumseagreen',
    'RBF':              'darkorange',
    'Polynomial (d=3)': 'mediumpurple',
    'Linear':           'sienna',
}

METRIC_LABELS = ['Alignment', 'Polarization', 'FSM', 'Complex Ratio']


def plot_kernel_metrics(ew, num_kernels: int) -> None:
    """Fig 31 — Heat-map of all 4 kernel quality metrics + natural MKL weights."""
    measures = ew.measures   # shape: (num_kernels, 4)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(5, num_kernels * 0.3 + 2)),
        gridspec_kw={'width_ratios': [3, 1]},
    )

    im = axes[0].imshow(measures, aspect='auto', cmap='RdYlGn')
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Weak Kernel Index')
    axes[0].set_title('Kernel Quality — all 4 metrics\n(used for extremality ordering)')
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(METRIC_LABELS, fontsize=8)
    axes[0].set_yticks(range(num_kernels))
    axes[0].set_yticklabels(range(num_kernels), fontsize=6)
    plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.04, label='Metric value')

    axes[1].barh(range(num_kernels), ew.w_natural, color='steelblue', alpha=0.8)
    axes[1].set_xlabel('Weight')
    axes[1].set_title('Natural MKL\nWeights')
    axes[1].set_yticks(range(num_kernels))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()

    fig.suptitle(
        f'Diverse Kernel Pool Analysis — {num_kernels} kernels '
        f'(poly + RBF + Laplacian), {MAX_FEATURES} max features',
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '31_mkl_kernel_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 31_mkl_kernel_metrics.png')


def plot_performance_comparison(results: dict) -> None:
    """Fig 32 — Grouped bar chart: Accuracy / F1 / AUC for all 6 strategies."""
    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]
    f1s   = [results[n]['f1']       for n in names]
    aucs  = [results[n]['auc']      for n in names]

    x, w = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, accs, w, label='Accuracy',   color='steelblue',      alpha=0.85)
    ax.bar(x,     f1s,  w, label='F1',         color='tomato',         alpha=0.85)
    ax.bar(x + w, aucs, w, label='AUC',        color='mediumseagreen', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(
        'SVM Classification Performance — Improved MKL vs Baselines\n'
        '(diverse pool: poly + RBF + Laplacian · all 4 metrics · C tuned by CV)'
    )
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='Random baseline')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '32_mkl_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 32_mkl_performance_comparison.png')


def plot_weight_distribution(ew) -> None:
    """Fig 33 — Natural vs anti-natural weight distribution and CDF."""
    order  = np.argsort(ew.w_natural)[::-1]
    w_nat  = ew.w_natural[order]
    w_anti = ew.w_anti_natural[order]
    x      = np.arange(len(w_nat))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, w_nat,  alpha=0.8, color='steelblue', label='Natural MKL')
    axes[0].bar(x, w_anti, alpha=0.6, color='tomato',    label='Anti-natural MKL')
    axes[0].set_xlabel('Weak Kernel (sorted by natural weight)')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('Kernel Weight Distribution')
    axes[0].legend()

    axes[1].plot(x, np.cumsum(w_nat),  color='steelblue', label='Natural MKL', linewidth=2)
    axes[1].plot(x, np.cumsum(w_anti), color='tomato',    label='Anti-natural', linewidth=2)
    axes[1].plot(x, np.cumsum(np.ones_like(w_nat) / len(w_nat)),
                 color='grey', linestyle='--', label='Uniform', linewidth=1)
    axes[1].set_xlabel('Top-k kernels included')
    axes[1].set_ylabel('Cumulative weight')
    axes[1].set_title('Cumulative Weight — Concentration of Extremal Kernels')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle('Extremality Kernel Weights — Crypto Price Direction (diverse pool)', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '33_mkl_weight_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 33_mkl_weight_distribution.png')


def plot_metrics_vs_num_kernels(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> None:
    """Fig 34 — AUC vs number of weak kernels (sweep, averaged over SWEEP_ITERS seeds)."""
    sweep: dict[str, list] = {k: [] for k in [
        'Natural MKL', 'Anti-Natural MKL', 'Uniform MKL',
        'RBF', 'Polynomial (d=3)', 'Linear',
    ]}

    for nk in KERNEL_SWEEP:
        per_name: dict[str, list] = {k: [] for k in sweep}
        for seed in range(SWEEP_ITERS):
            out = run_mkl_pipeline(
                X_train, X_test, y_train, y_test,
                num_kernels=nk, random_state=seed, tune_c=False,
            )
            for name, m in out['results'].items():
                per_name[name].append(m['auc'])
        for name in sweep:
            sweep[name].append(float(np.mean(per_name[name])))
        print(f'  [sweep] {nk} kernels done')

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, aucs in sweep.items():
        ax.plot(KERNEL_SWEEP, aucs, marker='o',
                label=name, color=PALETTE[name], linewidth=2)
    ax.set_xlabel('Number of Weak Kernels in Diverse Pool')
    ax.set_ylabel('Test AUC')
    ax.set_title(
        f'AUC vs Pool Size — {SWEEP_ITERS} seeds averaged\n'
        '(diverse pool: poly + RBF + Laplacian)'
    )
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '34_mkl_metrics_vs_kernels.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 34_mkl_metrics_vs_kernels.png')


def plot_confusion_matrices(results: dict, y_test: np.ndarray) -> None:
    """Fig 35 — 2×3 grid of confusion matrices for all kernel strategies."""
    names = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    y_test_01 = (y_test == 1).astype(int)

    for ax, name in zip(axes.flat, names):
        y_pred_01 = (results[name]['y_pred'] == 1).astype(int)
        cm   = confusion_matrix(y_test_01, y_pred_01)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Down (0)', 'Up (1)'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(
            f'{name}\nAUC={results[name]["auc"]:.3f}  C={results[name]["C_used"]}',
            fontsize=9,
        )

    fig.suptitle('Confusion Matrices — SVM with Improved MKL vs Baseline Kernels', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '35_mkl_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 35_mkl_confusion_matrices.png')


def plot_roc_curves(results: dict, y_test: np.ndarray) -> None:
    """Fig 36 — ROC curves for all kernel strategies on a single axes."""
    y_test_01 = (y_test == 1).astype(int)
    fig, ax   = plt.subplots(figsize=(8, 6))

    for name, m in results.items():
        RocCurveDisplay.from_predictions(
            y_test_01, m['y_score'],
            ax=ax,
            name=f'{name} (AUC={m["auc"]:.3f})',
            color=PALETTE[name],
        )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random baseline')
    ax.set_title('ROC Curves — Improved MKL vs Baseline Kernels\n(crypto price direction)')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '36_mkl_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 36_mkl_roc_curves.png')


# ── Report ─────────────────────────────────────────────────────────────────────

def generate_report(results: dict, ew, num_kernels: int) -> None:
    """Write a Markdown report summarising the improved MKL experiment."""
    ts        = datetime.now().strftime('%Y-%m-%d %H:%M')
    best_name = max(results, key=lambda k: results[k]['auc'])
    best      = results[best_name]

    lines = [
        '# SVM with Improved Multiple Kernel Learning (MKL) — Crypto ML Project',
        f'\n**Generated:** {ts}',
        '\n**Source:** adapted from https://github.com/maospina1041/extremality_mkl',
        '\n---\n',
        '## 1. Improvements over the baseline',
        '1. **Diverse kernel pool** — polynomial (d∈{2,3,4}) + RBF (4 γ values) '
        '+ Laplacian (4 γ values), all on random feature subsets.',
        '2. **All 4 metrics** — extremality ordering uses alignment, polarization, '
        'FSM, and complex_ratio simultaneously (vs. alignment + FSM only).',
        '3. **C selection by CV** — SVM regularisation C chosen by 3-fold stratified '
        'cross-validation inside the training set for each kernel strategy.',
        '\n---\n',
        '## 2. Experimental Setup',
        f'- **Task:** Binary classification — crypto price Up (+1) / Down (-1).',
        f'- **Features:** {len(BASE_FEATURES)} technical indicators.',
        f'- **Temporal split:** test = last {TEST_FRACTION:.0%} of dates.',
        f'- **Diverse pool:** {num_kernels} kernels, '
        f'up to {MAX_FEATURES} features/kernel.',
        f'- **Poly degrees:** {POLY_DEGREES} | **Gamma scales:** {GAMMA_SCALES}',
        f'- **Extremality exponent (n):** {MKL_EXPONENT}',
        f'- **C grid:** {C_GRID} (selected per-strategy by {CV_FOLDS}-fold CV)',
        '\n---\n',
        '## 3. Results\n',
        '| Kernel Strategy | Accuracy | F1 | AUC | C used |',
        '|-----------------|----------|----|-----|--------|',
    ]

    for name, m in results.items():
        marker = ' ← best' if name == best_name else ''
        lines.append(
            f'| {name}{marker} | {m["accuracy"]:.4f} | {m["f1"]:.4f} '
            f'| {m["auc"]:.4f} | {m["C_used"]} |'
        )

    lines += [
        f'\n### Best strategy: **{best_name}**',
        f'- Accuracy : {best["accuracy"]:.4f}',
        f'- F1       : {best["f1"]:.4f}',
        f'- AUC      : {best["auc"]:.4f}',
        f'- C        : {best["C_used"]}',
        '\n---\n',
        '## 4. Top 3 extremally-best kernels (Natural MKL)\n',
        '| Rank | Kernel # | Weight | Alignment | Polarization | FSM | Complex Ratio |',
        '|------|----------|--------|-----------|-------------|-----|---------------|',
    ]

    top3 = np.argsort(ew.w_natural)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        m_row = ew.measures[idx]
        lines.append(
            f'| {rank} | #{idx:02d} | {ew.w_natural[idx]:.4f} '
            f'| {m_row[0]:.4f} | {m_row[1]:.4f} | {m_row[2]:.4f} | {m_row[3]:.4f} |'
        )

    lines += [
        '\n---\n',
        '## 5. Interpretation',
        '- Kernels with high **alignment** have geometry closely matching the class structure.',
        '- High **FSM** indicates strong intra-class cohesion (small within-class spread, '
        'large between-class gap).',
        '- The **anti-natural MKL** should consistently underperform, validating the '
        'extremality ordering.',
        '- **C tuning** removes the shared-C confound between kernel strategies.',
        '\n---\n',
        '## 6. Figures',
        '- `31_mkl_kernel_metrics.png` — Heat-map of all 4 metrics + natural weights.',
        '- `32_mkl_performance_comparison.png` — Acc / F1 / AUC grouped bars.',
        '- `33_mkl_weight_distribution.png` — Weight distribution and CDF.',
        '- `34_mkl_metrics_vs_kernels.png` — AUC vs pool size sweep.',
        '- `35_mkl_confusion_matrices.png` — 2×3 confusion matrices.',
        '- `36_mkl_roc_curves.png` — ROC curves for all 6 strategies.',
        '\n---\n',
        '## 7. Related scripts (heart disease validation)',
        '- `svm_kernels/base/heart_base.py`     — baseline MKL on heart dataset.',
        '- `svm_kernels/improved/heart_improved.py` — improved MKL on heart dataset.',
        '- `svm_kernels/comparison.py`           — baseline vs improved comparison.',
    ]

    out = REPORTS_DIR / 'svm_mkl_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('[ok] svm_mkl_report.md')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print('[svm_mkl] Loading and splitting crypto features ...')
    X_train, X_test, y_train, y_test = load_data()
    print(f'  Features: {X_train.shape[1]}  |  '
          f'Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}')

    print(f'\n[svm_mkl] Running improved MKL pipeline '
          f'({NUM_KERNELS} diverse kernels, all 4 metrics, C by CV) ...')
    out = run_mkl_pipeline(X_train, X_test, y_train, y_test,
                           num_kernels=NUM_KERNELS, random_state=RANDOM_STATE,
                           tune_c=True)

    results = out['results']
    ew      = out['ew']

    print('\n[svm_mkl] Generating figures ...')
    plot_kernel_metrics(ew, NUM_KERNELS)
    plot_performance_comparison(results)
    plot_weight_distribution(ew)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)

    print('\n[svm_mkl] Running pool-size sweep (a few minutes) ...')
    plot_metrics_vs_num_kernels(X_train, X_test, y_train, y_test)

    print('\n[svm_mkl] Generating report ...')
    generate_report(results, ew, NUM_KERNELS)

    print('\n[done] svm_mkl pipeline completed.')
    print(f'       Results in: {REPORTS_DIR}')


if __name__ == '__main__':
    main()
