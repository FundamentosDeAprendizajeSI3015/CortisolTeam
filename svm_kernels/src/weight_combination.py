"""
Weighted linear combination utilities for kernel fusion.

Adapted from https://github.com/maospina1041/extremality_mkl
Applied to the Crypto ML Project (CortisolTeam).

After the extremality ordering ranks each weak kernel by quality, these
weights determine how strongly each kernel contributes to the final
combined (Gram) matrix that is fed to the SVM.
"""

import numpy as np


def compute_weights(scores: np.ndarray, exponent: int = 1) -> np.ndarray:
    """Convert raw kernel scores into a normalised weight vector.

    The transformation applies two successive L1-normalisations with a
    power in between, which sharpens the distribution: higher-ranked
    kernels receive disproportionately larger weights.

    Steps:
        1. Normalise scores to [0, 1]   →  x = scores / sum(scores)
        2. Raise to the power            →  x = x ** exponent
        3. Re-normalise                  →  x = x / sum(x)

    Args:
        scores:   1-D array of non-negative kernel quality scores.
        exponent: Sharpening exponent (n ≥ 1).  n=1 → proportional
                  weights; n>1 → winner-takes-most behaviour.

    Returns:
        1-D weight array of the same length as `scores`, summing to 1.
    """
    x = scores / scores.sum()
    x = x ** exponent
    x = x / x.sum()
    return x


def combine_kernels(KL: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Build a single Gram matrix as a weighted sum of kernel matrices.

    Args:
        KL:      Array of shape (num_kernels, n, n).
        weights: 1-D weight array of length num_kernels, summing to 1.

    Returns:
        Combined Gram matrix of shape (n, n).
    """
    # einsum 'ijk,i->jk' efficiently computes the weighted sum
    return np.einsum('ijk,i->jk', KL, weights)
