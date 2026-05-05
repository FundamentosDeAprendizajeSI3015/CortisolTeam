"""
Kernel quality metrics used to evaluate and rank weak kernels.

Adapted from https://github.com/maospina1041/extremality_mkl
Applied to the Crypto ML Project (CortisolTeam).

Each metric measures how well a kernel matrix separates the two classes
(price goes up = +1 / price goes down = -1).  Higher values are better
for alignment, polarization and FSM; lower is better for complex_ratio.

Backward-compatible aliases for the original extremality_mkl naming
convention are exported at the bottom of this file so that the heart-
disease baseline scripts (svm_kernels/base/ and svm_kernels/improved/)
continue to work without modification.
"""

import numpy as np


# ── Core metric functions ───────────────────────────────────────────────────

def complex_ratio(K: np.ndarray, y: np.ndarray = None) -> float:
    """Return the trace of the kernel matrix.

    A large trace means the kernel maps data far from the origin, which
    can indicate overfitting.  Used as a complexity penalty (direction: -1).

    Args:
        K: Square kernel matrix of shape (n, n).
        y: Ignored — kept for a uniform metric signature across all functions.

    Returns:
        Scalar trace value.
    """
    return float(np.trace(K))


def _ideal_kernel(y: np.ndarray) -> np.ndarray:
    """Build the ideal (oracle) kernel from class labels.

    The ideal kernel assigns +1 to same-class pairs and -1 to
    cross-class pairs, representing perfect class separation.

    Args:
        y: Label array of shape (n,) with values in {-1, +1}.

    Returns:
        Ideal kernel matrix of shape (n, n).
    """
    K_ideal = np.equal.outer(y, y).astype(float)
    # Replace 0 entries (cross-class pairs) with -1
    K_ideal[K_ideal == 0] = -1.0
    return K_ideal


def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Compute the Frobenius inner product alignment between K and the ideal kernel.

    Alignment measures how closely the kernel geometry matches the class
    structure.  Values in (0, 1]; higher is better.

    Formula:
        A(K, y) = <K, K_ideal>_F / (||K||_F * n)

    Args:
        K: Kernel matrix of shape (n, n).
        y: Label array of shape (n,) with values in {-1, +1}.

    Returns:
        Scalar alignment score.
    """
    K_ideal = _ideal_kernel(y)
    numerator = np.trace(K.T @ K_ideal)
    denominator = np.linalg.norm(K, 'fro') * len(y)
    return float(numerator / denominator)


def kernel_polarization(K: np.ndarray, y: np.ndarray) -> float:
    """Compute kernel polarization: total margin between class pairs.

    For each pair (i, j), polarization measures how much the kernel
    pushes same-class points together and cross-class points apart.
    Higher is better.

    Args:
        K: Kernel matrix of shape (n, n).
        y: Label array of shape (n,) with values in {-1, +1}.

    Returns:
        Scalar polarization score.
    """
    n = len(y)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Margin contribution of the pair (i, j)
            val = -y[i] * y[j] * (K[i, i] + K[j, j] - 2 * K[i, j])
            A[i, j] = val
            A[j, i] = val  # symmetric

    return float(np.sum(A))


def fsm(K: np.ndarray, y: np.ndarray) -> float:
    """Compute the Feature Space Measure (FSM).

    FSM quantifies intra-class cohesion relative to inter-class dispersion
    in the kernel-induced feature space.  Higher is better.

    Args:
        K: Kernel matrix of shape (n, n).
        y: Label array of shape (n,) with values in {-1, +1}.

    Returns:
        Scalar FSM score.
    """
    neg_mask = y == -1
    pos_mask = y == 1
    n_neg = int(neg_mask.sum())
    n_pos = int(pos_mask.sum())

    if n_neg == 0 or n_pos == 0:
        return 0.0

    # Per-sample mean similarity within/across classes
    d_i = K[np.ix_(neg_mask, neg_mask)].sum(axis=1) / n_neg   # neg intra
    a_i = K[np.ix_(pos_mask, pos_mask)].sum(axis=1) / n_pos   # pos intra
    c_i = K[np.ix_(neg_mask, pos_mask)].sum(axis=1) / n_pos   # neg→pos inter
    b_i = K[np.ix_(pos_mask, neg_mask)].sum(axis=1) / n_neg   # pos→neg inter

    # Global averages
    A = a_i.sum() / n_pos
    B = b_i.sum() / n_pos
    C = c_i.sum() / n_neg
    D = d_i.sum() / n_neg

    phi_sq = A + D - B - C  # normalisation factor

    if phi_sq <= 0:
        return 0.0

    aux_1 = np.sum((b_i - a_i + A - B) ** 2) / (phi_sq * max(n_pos - 1, 1))
    aux_2 = np.sum((c_i - d_i + D - C) ** 2) / (phi_sq * max(n_neg - 1, 1))

    return float((np.sqrt(aux_1) + np.sqrt(aux_2)) / np.sqrt(phi_sq))


# ── Backward-compatible aliases ─────────────────────────────────────────────
# The original extremality_mkl repo used slightly different names.
# These aliases let the heart-disease baseline scripts (svm_kernels/base/
# and svm_kernels/improved/) import without any modification.

FSM             = fsm               # original used uppercase FSM
kernel_aligment = kernel_alignment  # original had a typo: 'aligment'
ideal_kernel    = _ideal_kernel     # original had no leading underscore
