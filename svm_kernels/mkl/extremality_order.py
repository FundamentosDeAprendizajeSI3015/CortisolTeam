"""
Extremality ordering of kernel matrices.

Adapted from https://github.com/maospina1041/extremality_mkl
Applied to the Crypto ML Project (CortisolTeam).

The extremality order is a multi-criteria ranking that avoids the
scalarisation bias of a single weighted sum.  Given a matrix of kernel
quality scores (one row per kernel, one column per metric), it uses a
rotation in metric space to determine which kernels dominate all others
simultaneously across all metrics.

References:
    Maospina (2024). extremality_mkl.
    https://github.com/maospina1041/extremality_mkl
"""

import numpy as np


# ── Gram-Schmidt orthogonalisation ─────────────────────────────────────────

def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonalise the columns of A via the classical Gram-Schmidt process.

    Args:
        A: Input matrix of shape (m, n).

    Returns:
        (Q, R) where Q is orthonormal (m×n) and R is upper-triangular (n×n).
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


# ── Rotation matrix from a direction vector ────────────────────────────────

def build_rotation_matrix(u: np.ndarray) -> np.ndarray:
    """Build a rotation matrix that aligns the first canonical basis vector
    with the direction of u.

    The rotation is computed by Gram-Schmidt orthogonalising two frames:
        - M_u: a frame whose first column is u / ||u||.
        - M_e: the standard frame whose first column is 1/√n · 1.

    The rotation R = Q_e · Q_u^T maps the u-frame to the standard frame.

    Args:
        u: Direction vector of length n (typically the metric direction array).

    Returns:
        Rotation matrix R of shape (n, n).
    """
    n = len(u)
    identity = np.eye(n)

    # Frame aligned with u
    M_u = np.sign(u).reshape(n, 1) * identity
    M_u[:, 0] = u / np.linalg.norm(u, 2)

    # Standard frame
    M_e = identity.copy()
    M_e[:, 0] = np.ones(n) / np.sqrt(n)

    Q_u, _ = gram_schmidt(M_u)
    Q_e, _ = gram_schmidt(M_e)

    return Q_e @ Q_u.T


# ── Extremality order ──────────────────────────────────────────────────────

def extremality_order(
    kernel_metrics: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Rank kernels by their extremality in the rotated metric space.

    A kernel k is assigned rank r equal to the number of other kernels
    that dominate it (i.e. are ≥ k in *every* rotated metric dimension).
    Rank 1 means no kernel dominates it → it is extremal (best).

    Algorithm:
        1. Rotate the metric matrix using build_rotation_matrix(direction).
        2. For each kernel i, count how many kernels j satisfy
           rotated[j] ≥ rotated[i] in all dimensions simultaneously.

    Args:
        kernel_metrics: Matrix of shape (num_kernels, num_metrics).
                        Each row is a kernel; each column is a metric value.
        direction:      Sign array of length num_metrics (+1 = higher is
                        better, -1 = lower is better).

    Returns:
        1-D array of length num_kernels with integer rank scores.
        Lower rank ↔ more extremal (better).
    """
    R = build_rotation_matrix(direction)
    # Project each kernel's metric row into the rotated space
    rotated = (R @ kernel_metrics.T).T   # shape: (num_kernels, num_metrics)

    num_kernels = kernel_metrics.shape[0]
    ranks = np.zeros(num_kernels, dtype=float)

    for i in range(num_kernels):
        # Count kernels that dominate kernel i across ALL rotated dimensions
        ranks[i] = np.sum(np.all(rotated >= rotated[i], axis=1))

    return ranks
