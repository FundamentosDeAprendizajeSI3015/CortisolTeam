"""
Extremality-based kernel weighting (MKL core).

Adapted from https://github.com/maospina1041/extremality_mkl
Applied to the Crypto ML Project (CortisolTeam).

This module ties together:
    1. Kernel quality metrics (alignment, polarization, FSM, complex_ratio).
    2. Extremality ordering to rank kernels without a scalar aggregation.
    3. Soft weight assignment so that the final SVM Gram matrix is a
       convex combination of the weak kernels.

Two complementary weight sets are produced:
    - w_natural:     favours kernels that are extremal in the "good"
                     direction (high alignment, high FSM → better SVM margin).
    - w_anti_natural: favours kernels that are extremal in the opposite
                     direction (intentionally "bad" kernels).  This serves
                     as a built-in ablation baseline.
"""

import numpy as np

from svm_kernels.src.kernel_metrics import (
    kernel_alignment,
    kernel_polarization,
    fsm,
    complex_ratio,
)
from svm_kernels.src.weight_combination import compute_weights
from svm_kernels.mkl.extremality_order import extremality_order


# ── Metric registry ────────────────────────────────────────────────────────

# Each entry: name → (function, direction)
# direction = +1 means "higher is better"; -1 means "lower is better".
ALL_METRICS: dict[str, tuple] = {
    "alignment":     (kernel_alignment,     +1),
    "polarization":  (kernel_polarization,  +1),
    "fsm":           (fsm,                  +1),
    "complex_ratio": (complex_ratio,        -1),
}

# Subset used for kernel weighting (alignment + FSM give the best signal
# for binary crypto classification without the noise of polarization).
WEIGHT_METRICS: dict[str, tuple] = {
    "alignment": (kernel_alignment, +1),
    "fsm":       (fsm,              +1),
}


# ── Metric computation ─────────────────────────────────────────────────────

def compute_kernel_metrics(
    KL_train: np.ndarray,
    y_train: np.ndarray,
    metrics: dict[str, tuple] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate each weak kernel against each quality metric.

    Args:
        KL_train: Array of shape (num_kernels, n_train, n_train).
        y_train:  Label array of shape (n_train,) with values in {-1, +1}.
        metrics:  Metric registry to use.  Defaults to ALL_METRICS.

    Returns:
        (measures, directions):
            measures   — float matrix of shape (num_kernels, num_metrics).
            directions — int array of length num_metrics (+1 or -1).
    """
    if metrics is None:
        metrics = ALL_METRICS

    num_kernels = KL_train.shape[0]
    num_metrics = len(metrics)
    measures = np.zeros((num_kernels, num_metrics))

    for i, (name, (fn, _)) in enumerate(metrics.items()):
        for k in range(num_kernels):
            measures[k, i] = fn(KL_train[k], y_train)

    directions = np.array([sign for _, (_, sign) in metrics.items()], dtype=float)
    return measures, directions


# ── Result container ───────────────────────────────────────────────────────

class ExtremalityWeights:
    """Container for the two complementary kernel weight vectors.

    Attributes:
        w_natural:      Weights that boost extremally-good kernels.
        w_anti_natural: Weights that boost extremally-bad kernels
                        (useful as an ablation baseline).
        measures:       Raw metric matrix (num_kernels × num_metrics).
        directions:     Metric direction signs.
    """

    def __init__(
        self,
        w_natural: np.ndarray,
        w_anti_natural: np.ndarray,
        measures: np.ndarray,
        directions: np.ndarray,
    ) -> None:
        self.w_natural = w_natural
        self.w_anti_natural = w_anti_natural
        self.measures = measures
        self.directions = directions


# ── Main API ───────────────────────────────────────────────────────────────

def compute_extremality_weights(
    KL_train: np.ndarray,
    y_train: np.ndarray,
    metrics: dict[str, tuple] | None = None,
    exponent: int = 1,
) -> ExtremalityWeights:
    """Compute natural and anti-natural kernel weights via extremality ordering.

    Pipeline:
        1. Evaluate each weak kernel on the chosen quality metrics.
        2. Rank kernels using the extremality order (multi-criteria Pareto
           dominance in a rotated metric space).
        3. Convert ranks to soft weights with compute_weights().

    Natural weights (w_natural):
        Order in the "+direction" (good kernels rank first).
        Weight = high for kernels with low dominance count (= extremal best).

    Anti-natural weights (w_anti_natural):
        Order in the "-direction" (bad kernels rank first).
        Weight = high for kernels with *high* dominance count in inverted space.

    Args:
        KL_train: Weak kernel stack, shape (num_kernels, n_train, n_train).
        y_train:  Binary labels in {-1, +1}, shape (n_train,).
        metrics:  Metric registry.  Defaults to WEIGHT_METRICS
                  (alignment + FSM), which proved most informative for
                  the crypto price-direction classification task.
        exponent: Sharpening exponent for compute_weights().

    Returns:
        ExtremalityWeights instance with w_natural and w_anti_natural.
    """
    if metrics is None:
        metrics = WEIGHT_METRICS

    measures, directions = compute_kernel_metrics(KL_train, y_train, metrics)

    # Natural order: extremal kernels in the "good" direction get low rank →
    # invert rank so that best kernels receive highest weight.
    ranks_natural = extremality_order(measures, directions)
    n_kernels = len(ranks_natural)
    w_natural = compute_weights(n_kernels - ranks_natural + 1, exponent)

    # Anti-natural order: best kernels in the flipped direction
    ranks_anti = extremality_order(measures, -directions)
    w_anti_natural = compute_weights(ranks_anti + 1, exponent)

    return ExtremalityWeights(w_natural, w_anti_natural, measures, directions)
