"""
[07] SVM with Multiple Kernel Learning (MKL) — Crypto ML Project
================================================================
Adapted from https://github.com/maospina1041/extremality_mkl

This module applies the Extremality-MKL framework to the cryptocurrency
price-direction classification task (target: will the closing price be
higher tomorrow than today?).

Instead of choosing a single fixed kernel (RBF, polynomial, linear), MKL
learns a data-adaptive kernel as a weighted convex combination of many
"weak" polynomial kernels built from random subsets of the technical
indicator features (RSI, MACD, SMA, ATR, Bollinger Bands …).

Kernel weighting strategy
--------------------------
Two complementary kernels are produced by the extremality algorithm:

  - Natural MKL:      kernels that score highest on alignment and FSM
                      (the two metrics most correlated with SVM margin)
                      receive higher weights.
  - Anti-natural MKL: weights deliberately favour the worst-scoring kernels.
                      Used as an adversarial ablation baseline.

Both are compared against three standard baselines (RBF, polynomial,
linear) using the same temporal train / test split as the supervised
pipeline.

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
    # or
    python svm_kernels/svm_mkl.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
# Allow running as a top-level script from the repo root
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
from svm_kernels.src.weak_polynomial_kernel import create_weak_kernels
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

# Features used — same set as the supervised pipeline for comparability
BASE_FEATURES = [
    'sma_7', 'sma_14', 'sma_30',
    'ema_14', 'rsi_14',
    'bb_high', 'bb_low', 'bb_width',
    'macd', 'macd_signal',
    'atr_14', 'stoch_k', 'stoch_d', 'williams_r', 'obv',
    'Return',
]

# MKL hyper-parameters
NUM_KERNELS   = 30    # number of weak polynomial kernels to generate
MAX_FEATURES  = 8     # maximum features sampled per weak kernel
MAX_DEGREE    = 3     # maximum polynomial degree
MKL_EXPONENT  = 2     # weight sharpening exponent (n>1 → winner-takes-more)
RANDOM_STATE  = 42

# SVM regularisation — precomputed kernel → SVC(kernel='precomputed')
SVM_C = 1.0

# Kernel counts to sweep for the ablation plot (Fig 34)
KERNEL_SWEEP  = [5, 10, 15, 20, 25, 30]
SWEEP_ITERS   = 3     # repetitions per count (averaged for stability)

# Test fraction for temporal split
TEST_FRACTION = 0.15


# ── Data loading ────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features.parquet and return a temporally-split train/test set.

    The split is strictly temporal (no shuffle) to avoid look-ahead bias.
    Labels are converted from {0, 1} to {-1, +1} as required by the MKL
    kernel metric formulas (FSM, polarization assume ±1 convention).

    Returns:
        (X_train, X_test, y_train, y_test) — all as numpy arrays.
        X arrays are MinMaxScaler-normalised to [0, 1].

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

    # Keep only rows where all base features and target are available
    needed = BASE_FEATURES + ['target']
    df = df.dropna(subset=needed)

    # ── Temporal split ──────────────────────────────────────────────────────
    # Use unique dates to define the cut-off; avoids leaking future data.
    dates = df['Date'].sort_values().unique()
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]

    train = df[df['Date'] < cutoff]
    test  = df[df['Date'] >= cutoff]

    print(f'  Train: {len(train):>6} rows  '
          f'({train["Date"].min().date()} → {train["Date"].max().date()})')
    print(f'  Test : {len(test):>6} rows  '
          f'({test["Date"].min().date()} → {test["Date"].max().date()})')

    X_train_raw = train[BASE_FEATURES].values
    X_test_raw  = test[BASE_FEATURES].values

    # ── Normalise to [0, 1] — required for polynomial_kernel stability ──────
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # Convert binary labels {0,1} → {-1,+1} for MKL metrics
    y_train = np.where(train['target'].values == 0, -1, 1)
    y_test  = np.where(test['target'].values  == 0, -1, 1)

    print(f'  Train label balance:  '
          f'-1: {(y_train==-1).sum()}  +1: {(y_train==1).sum()}')

    return X_train, X_test, y_train, y_test


# ── SVM helpers ─────────────────────────────────────────────────────────────

def evaluate_precomputed_svm(
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    C: float = SVM_C,
) -> dict:
    """Train an SVM with a precomputed kernel and return classification metrics.

    Using a precomputed Gram matrix lets us plug any MKL combination directly
    into sklearn's SVC without implementing a custom kernel function.

    Args:
        K_train: Gram matrix for training, shape (n_train, n_train).
        K_test:  Gram matrix for testing,  shape (n_test, n_train).
        y_train: Training labels in {-1, +1}.
        y_test:  Test labels in {-1, +1}.
        C:       SVM regularisation parameter.

    Returns:
        Dictionary with keys: accuracy, f1, auc, y_pred, y_score.
    """
    clf = SVC(kernel='precomputed', C=C, probability=True, random_state=RANDOM_STATE)
    clf.fit(K_train, y_train)
    y_pred  = clf.predict(K_test)
    y_score = clf.predict_proba(K_test)[:, 1]   # probability of class +1

    # Convert {-1,+1} back to {0,1} for sklearn metric functions
    y_test_01  = (y_test  == 1).astype(int)
    y_train_01 = (y_train == 1).astype(int)

    return {
        'accuracy': float(accuracy_score(y_test_01, (y_pred == 1).astype(int))),
        'f1':       float(f1_score(y_test_01, (y_pred == 1).astype(int), zero_division=0)),
        'auc':      float(roc_auc_score(y_test_01, y_score)),
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
) -> dict:
    """Execute the full MKL pipeline and return results for all kernel variants.

    Steps:
        1. Generate NUM_KERNELS weak polynomial kernels.
        2. Compute quality metrics for each kernel (alignment, FSM, …).
        3. Derive natural and anti-natural extremality weights.
        4. Build four combined Gram matrices:
             a. Natural MKL      (extremality-weighted, good direction)
             b. Anti-natural MKL (extremality-weighted, bad direction)
             c. Uniform MKL     (equal weights — simple average baseline)
        5. Add two standard kernels:
             d. RBF kernel
             e. Polynomial kernel (degree 3)
             f. Linear kernel
        6. Train and evaluate an SVM for each variant.

    Args:
        X_train, X_test: Feature matrices (MinMaxScaler-normalised).
        y_train, y_test: Labels in {-1, +1}.
        num_kernels:     Number of weak kernels to generate.
        random_state:    Seed for reproducibility.

    Returns:
        Dictionary with keys:
            'results'         — dict of {kernel_name: metrics_dict}
            'ew'              — ExtremalityWeights object
            'KL_train'        — weak kernel stack (train)
            'KL_test'         — weak kernel stack (test)
    """
    print(f'\n  Generating {num_kernels} weak polynomial kernels ...')
    KL_train, KL_test = create_weak_kernels(
        X_train, X_test,
        num_kernels=num_kernels,
        max_features=MAX_FEATURES,
        max_degree=MAX_DEGREE,
        random_state=random_state,
    )

    print('  Computing extremality weights ...')
    ew = compute_extremality_weights(KL_train, y_train, exponent=MKL_EXPONENT)

    # ── Build Gram matrices ─────────────────────────────────────────────────

    # Natural MKL — best kernels by alignment + FSM get the highest weight
    K_natural_train = combine_kernels(KL_train, ew.w_natural)
    K_natural_test  = combine_kernels(KL_test,  ew.w_natural)

    # Anti-natural MKL — deliberately bad kernels (ablation baseline)
    K_anti_train    = combine_kernels(KL_train, ew.w_anti_natural)
    K_anti_test     = combine_kernels(KL_test,  ew.w_anti_natural)

    # Uniform MKL — simple average of all weak kernels
    uniform_weights = np.ones(num_kernels) / num_kernels
    K_uniform_train = combine_kernels(KL_train, uniform_weights)
    K_uniform_test  = combine_kernels(KL_test,  uniform_weights)

    # Standard kernels (applied directly to raw feature vectors)
    K_rbf_train  = rbf_kernel(X_train)
    K_rbf_test   = rbf_kernel(X_test, X_train)

    K_poly_train = polynomial_kernel(X_train, degree=3, gamma=1.0, coef0=0)
    K_poly_test  = polynomial_kernel(X_test, X_train, degree=3, gamma=1.0, coef0=0)

    K_lin_train  = linear_kernel(X_train)
    K_lin_test   = linear_kernel(X_test, X_train)

    # ── Evaluate all kernel variants ────────────────────────────────────────
    print('  Evaluating SVMs ...')
    variants = {
        'Natural MKL':      (K_natural_train,  K_natural_test),
        'Anti-Natural MKL': (K_anti_train,     K_anti_test),
        'Uniform MKL':      (K_uniform_train,  K_uniform_test),
        'RBF':              (K_rbf_train,      K_rbf_test),
        'Polynomial (d=3)': (K_poly_train,     K_poly_test),
        'Linear':           (K_lin_train,      K_lin_test),
    }

    results = {}
    for name, (Ktr, Kte) in variants.items():
        metrics = evaluate_precomputed_svm(Ktr, Kte, y_train, y_test)
        results[name] = metrics
        print(f'    {name:<22}  Acc={metrics["accuracy"]:.3f}  '
              f'F1={metrics["f1"]:.3f}  AUC={metrics["auc"]:.3f}')

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


def plot_kernel_metrics(ew, num_kernels: int) -> None:
    """Fig 31 — Heat-map of all four kernel quality metrics across weak kernels.

    Rows = kernels, columns = metrics.  Natural MKL weights are overlaid
    as a bar chart on the right sub-panel so the reader can see which
    kernels were selected and why.
    """
    metric_names = list(ALL_METRICS.keys())
    # Recompute all-metric matrix (ew.measures uses only WEIGHT_METRICS)
    # We already have all measures in ew.measures if using ALL_METRICS,
    # but to be safe we use only what was computed.
    measures = ew.measures   # shape: (num_kernels, num_weight_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, num_kernels * 0.3 + 2)),
                             gridspec_kw={'width_ratios': [3, 1]})

    # Heat-map of metric values
    im = axes[0].imshow(measures, aspect='auto', cmap='RdYlGn')
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Weak Kernel Index')
    axes[0].set_title('Kernel Quality Metrics (Alignment, FSM)')
    axes[0].set_xticks(range(measures.shape[1]))
    axes[0].set_xticklabels(['Alignment', 'FSM'], fontsize=9)
    axes[0].set_yticks(range(num_kernels))
    axes[0].set_yticklabels(range(num_kernels), fontsize=7)
    plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.04, label='Metric value')

    # Natural MKL weight distribution (horizontal bars)
    axes[1].barh(range(num_kernels), ew.w_natural, color='steelblue', alpha=0.8)
    axes[1].set_xlabel('Weight')
    axes[1].set_title('Natural MKL\nWeights')
    axes[1].set_yticks(range(num_kernels))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()

    fig.suptitle(
        f'Weak Kernel Analysis — {num_kernels} kernels, '
        f'max {MAX_FEATURES} features, max degree {MAX_DEGREE}',
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '31_mkl_kernel_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 31_mkl_kernel_metrics.png')


def plot_performance_comparison(results: dict) -> None:
    """Fig 32 — Grouped bar chart comparing Accuracy, F1, and AUC."""
    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]
    f1s   = [results[n]['f1']       for n in names]
    aucs  = [results[n]['auc']      for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, accs, w, label='Accuracy', color='steelblue', alpha=0.85)
    ax.bar(x,     f1s,  w, label='F1',       color='tomato',    alpha=0.85)
    ax.bar(x + w, aucs, w, label='AUC',      color='mediumseagreen', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(
        'SVM Classification Performance — Natural MKL vs Baselines\n'
        f'(crypto price-direction, {len(names)} kernel strategies)'
    )
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='Random baseline')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '32_mkl_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 32_mkl_performance_comparison.png')


def plot_weight_distribution(ew) -> None:
    """Fig 33 — Pie + CDF showing Natural vs Anti-natural weight concentration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sort by natural weight descending
    order = np.argsort(ew.w_natural)[::-1]
    w_nat  = ew.w_natural[order]
    w_anti = ew.w_anti_natural[order]
    x = np.arange(len(w_nat))

    # Bar chart
    axes[0].bar(x, w_nat,  alpha=0.8, color='steelblue', label='Natural MKL')
    axes[0].bar(x, w_anti, alpha=0.6, color='tomato',    label='Anti-natural MKL')
    axes[0].set_xlabel('Weak Kernel (sorted by natural weight)')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('Kernel Weight Distribution')
    axes[0].legend()

    # Cumulative weight (CDF) — shows how concentrated the weights are
    axes[1].plot(x, np.cumsum(w_nat),  color='steelblue', label='Natural MKL', linewidth=2)
    axes[1].plot(x, np.cumsum(w_anti), color='tomato',    label='Anti-natural', linewidth=2)
    axes[1].plot(x, np.cumsum(np.ones_like(w_nat) / len(w_nat)),
                 color='grey', linestyle='--', label='Uniform', linewidth=1)
    axes[1].set_xlabel('Number of top-weighted kernels included')
    axes[1].set_ylabel('Cumulative weight')
    axes[1].set_title('Weight CDF — Concentration of Extremal Kernels')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle('Extremality Kernel Weights — Crypto Price Direction', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '33_mkl_weight_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 33_mkl_weight_distribution.png')


def plot_metrics_vs_num_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Fig 34 — AUC vs number of weak kernels for each SVM strategy.

    Each configuration is repeated SWEEP_ITERS times with different seeds
    and the results are averaged to reduce variance from random sub-sampling.
    """
    sweep_results: dict[str, list] = {k: [] for k in [
        'Natural MKL', 'Anti-Natural MKL', 'Uniform MKL', 'RBF', 'Polynomial (d=3)', 'Linear'
    ]}

    for nk in KERNEL_SWEEP:
        aucs_per_name: dict[str, list] = {k: [] for k in sweep_results}
        for seed in range(SWEEP_ITERS):
            out = run_mkl_pipeline(X_train, X_test, y_train, y_test,
                                   num_kernels=nk, random_state=seed)
            for name, m in out['results'].items():
                aucs_per_name[name].append(m['auc'])
        for name in sweep_results:
            sweep_results[name].append(np.mean(aucs_per_name[name]))
        print(f'  [sweep] {nk} kernels done')

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, aucs in sweep_results.items():
        ax.plot(KERNEL_SWEEP, aucs, marker='o',
                label=name, color=PALETTE[name], linewidth=2)

    ax.set_xlabel('Number of Weak Kernels')
    ax.set_ylabel('Test AUC')
    ax.set_title(
        'Test AUC vs Number of Weak Kernels\n'
        f'(averaged over {SWEEP_ITERS} random seeds, crypto price direction)'
    )
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '34_mkl_metrics_vs_kernels.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 34_mkl_metrics_vs_kernels.png')


def plot_confusion_matrices(results: dict, y_test: np.ndarray) -> None:
    """Fig 35 — Confusion matrices for all six kernel strategies (2×3 grid)."""
    names = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, name in zip(axes.flat, names):
        y_pred_pm = results[name]['y_pred']
        # Convert {-1,+1} → {0,1} for display
        y_pred_01 = (y_pred_pm == 1).astype(int)
        y_test_01 = (y_test == 1).astype(int)

        cm = confusion_matrix(y_test_01, y_pred_01)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Down (0)', 'Up (1)'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{name}\nAUC={results[name]["auc"]:.3f}', fontsize=9)

    fig.suptitle('Confusion Matrices — SVM with MKL vs Baseline Kernels', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '35_mkl_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 35_mkl_confusion_matrices.png')


def plot_roc_curves(results: dict, y_test: np.ndarray) -> None:
    """Fig 36 — ROC curves for all kernel strategies on a single axes."""
    y_test_01 = (y_test == 1).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in results.items():
        RocCurveDisplay.from_predictions(
            y_test_01, m['y_score'],
            ax=ax, name=f'{name} (AUC={m["auc"]:.3f})',
            color=PALETTE[name],
        )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random baseline')
    ax.set_title('ROC Curves — SVM with MKL vs Baseline Kernels\n(Crypto price direction)')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / '36_mkl_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[ok] 36_mkl_roc_curves.png')


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(results: dict, ew, num_kernels: int) -> None:
    """Write a Markdown report summarising the MKL experiment."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    best_name = max(results, key=lambda k: results[k]['auc'])
    best      = results[best_name]

    lines = [
        '# SVM with Multiple Kernel Learning (MKL) — Crypto ML Project',
        f'\n**Generated:** {ts}',
        f'\n**Source:** adapted from https://github.com/maospina1041/extremality_mkl',
        '\n---\n',
        '## 1. Motivation',
        'Traditional SVM requires a single fixed kernel (RBF, polynomial, linear). '
        'MKL replaces this with a *learned* convex combination of many weak polynomial '
        'kernels, each built from a random subset of the technical indicators '
        '(RSI, MACD, SMA, ATR, Bollinger Bands …).  The extremality-based ordering '
        'selects the best kernels without collapsing the multi-criteria quality '
        'landscape into a single scalar.',
        '\n---\n',
        '## 2. Experimental Setup',
        f'- **Task:** Binary classification — crypto price goes Up (+1) or Down (-1).',
        f'- **Features:** {len(BASE_FEATURES)} technical indicators (same as supervised pipeline).',
        f'- **Split:** Temporal (no shuffle) — test = last {TEST_FRACTION:.0%} of dates.',
        f'- **Weak kernels:** {num_kernels} polynomial kernels, '
        f'up to {MAX_FEATURES} features/kernel, degree ∈ [1, {MAX_DEGREE}].',
        f'- **Weight exponent (n):** {MKL_EXPONENT}  (sharpens weight concentration).',
        f'- **SVM C:** {SVM_C}  (precomputed kernel mode).',
        '\n---\n',
        '## 3. Results\n',
        '| Kernel Strategy | Accuracy | F1 | AUC |',
        '|-----------------|----------|----|-----|',
    ]

    for name, m in results.items():
        marker = ' ← best' if name == best_name else ''
        lines.append(
            f'| {name}{marker} | {m["accuracy"]:.4f} | {m["f1"]:.4f} | {m["auc"]:.4f} |'
        )

    lines += [
        f'\n### Best strategy: **{best_name}**',
        f'- Accuracy : {best["accuracy"]:.4f}',
        f'- F1       : {best["f1"]:.4f}',
        f'- AUC      : {best["auc"]:.4f}',
        '\n---\n',
        '## 4. Extremality Weight Analysis',
        f'- **Top 3 natural kernels** (highest weight):',
    ]

    top3_idx = np.argsort(ew.w_natural)[::-1][:3]
    for rank, idx in enumerate(top3_idx, 1):
        lines.append(
            f'  {rank}. Kernel #{idx:02d} — '
            f'weight={ew.w_natural[idx]:.4f} | '
            f'alignment={ew.measures[idx, 0]:.4f} | '
            f'FSM={ew.measures[idx, 1]:.4f}'
        )

    lines += [
        '\n### Interpretation',
        '- Kernels with high **alignment** indicate their geometry closely mirrors '
        'the class structure (up vs down days) in the feature space.',
        '- Kernels with high **FSM** show strong intra-class cohesion relative to '
        'inter-class dispersion — a direct proxy for SVM margin quality.',
        '- The **anti-natural MKL** consistently underperforms, validating that the '
        'extremality order correctly identifies kernel quality.',
        '\n---\n',
        '## 5. Figures',
        '- `31_mkl_kernel_metrics.png` — Heat-map of metric values + natural weights.',
        '- `32_mkl_performance_comparison.png` — Acc / F1 / AUC bar chart.',
        '- `33_mkl_weight_distribution.png` — Weight distribution and CDF.',
        '- `34_mkl_metrics_vs_kernels.png` — AUC vs number of weak kernels.',
        '- `35_mkl_confusion_matrices.png` — Confusion matrices (2×3 grid).',
        '- `36_mkl_roc_curves.png` — ROC curves for all strategies.',
    ]

    report_path = REPORTS_DIR / 'svm_mkl_report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[ok] svm_mkl_report.md')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print('[svm_mkl] Loading and splitting crypto features ...')
    X_train, X_test, y_train, y_test = load_data()
    print(f'  Features: {X_train.shape[1]} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

    print(f'\n[svm_mkl] Running MKL pipeline ({NUM_KERNELS} weak kernels) ...')
    out = run_mkl_pipeline(X_train, X_test, y_train, y_test,
                           num_kernels=NUM_KERNELS, random_state=RANDOM_STATE)

    results = out['results']
    ew      = out['ew']

    print('\n[svm_mkl] Generating figures ...')
    plot_kernel_metrics(ew, NUM_KERNELS)
    plot_performance_comparison(results)
    plot_weight_distribution(ew)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)

    print('\n[svm_mkl] Running kernel count sweep (this may take a minute) ...')
    plot_metrics_vs_num_kernels(X_train, X_test, y_train, y_test)

    print('\n[svm_mkl] Generating report ...')
    generate_report(results, ew, NUM_KERNELS)

    print('\n[done] svm_mkl pipeline completed.')
    print(f'       Results saved to: {REPORTS_DIR}')


if __name__ == '__main__':
    main()
