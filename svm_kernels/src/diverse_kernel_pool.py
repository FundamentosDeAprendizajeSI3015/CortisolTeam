"""
Diversified weak kernel pool: polynomial + RBF + Laplacian.

This module extends the original weak_polynomial_kernel approach by mixing
three kernel families, an improvement introduced in
svm_kernels/improved/heart_improved.py and generalised here for the
Crypto ML Project (CortisolTeam).

Motivation
----------
Polynomial kernels on random feature subsets already provide diversity, but
they can miss global similarity patterns that radial kernels capture well.
By randomly drawing the kernel *type* (polynomial / RBF / Laplacian) as well
as the feature subset and hyper-parameters, the pool spans a richer region of
the kernel Hilbert space.  The extremality ordering then selects the most
informative combination for the crypto price-direction task.

Key difference vs weak_polynomial_kernel.py
--------------------------------------------
- Kernel type is sampled: polynomial (various degrees),
  RBF (various gamma values), or Laplacian (various gamma values).
- gamma values are scaled relative to 1/n_features so they remain
  sensible across datasets of different dimensionality.
"""

import numpy as np
from sklearn.metrics.pairwise import (
    polynomial_kernel,
    rbf_kernel,
    laplacian_kernel,
)


def create_diverse_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
    num_kernels: int = 20,
    max_features: int = 8,
    poly_degrees: tuple[int, ...] = (2, 3, 4),
    gamma_scales: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
    """Generate a diversified pool of weak kernels.

    Each kernel is built from:
        - A random subset of feature columns (with replacement).
        - A randomly selected kernel family (polynomial / RBF / Laplacian).
        - A randomly selected hyper-parameter for that family
          (degree for polynomial; gamma for radial kernels).

    Gamma values are specified as multiples of 1/n_features so they stay
    proportional to the feature space dimension, following the convention
    from heart_improved.py.

    Args:
        X_train:      Training feature matrix (n_train, n_features).
                      Must be normalised (e.g. with MinMaxScaler) before call.
        X_test:       Test feature matrix (n_test, n_features).  Optional.
        num_kernels:  Total number of weak kernels to generate.
        max_features: Maximum number of features selected per kernel.
        poly_degrees: Tuple of polynomial degrees to sample from.
        gamma_scales: Gamma values expressed as multiples of 1/n_features.
        random_state: Seed for reproducibility.

    Returns:
        If X_test is None:
            KL_train — array of shape (num_kernels, n_train, n_train).
        If X_test is provided:
            (KL_train, KL_test) where KL_test has shape
            (num_kernels, n_test, n_train).
    """
    rng = np.random.default_rng(random_state)
    n_train, n_features = X_train.shape

    # Absolute gamma values scaled to the feature space dimensionality
    gammas = [s / n_features for s in gamma_scales]

    KL_train = []
    KL_test  = [] if X_test is not None else None

    for _ in range(num_kernels):
        # Random feature subset (with replacement for extra diversity)
        k = int(rng.integers(1, max_features + 1))
        col_idx = rng.integers(0, n_features, size=k)
        X1 = X_train[:, col_idx]

        # Randomly pick kernel family
        family = rng.choice(['poly', 'rbf', 'laplacian'])

        if family == 'poly':
            degree = int(rng.choice(poly_degrees))
            Ktr = polynomial_kernel(X1, degree=degree, gamma=1.0, coef0=0)
            if X_test is not None:
                X2  = X_test[:, col_idx]
                Kte = polynomial_kernel(X2, X1, degree=degree, gamma=1.0, coef0=0)

        elif family == 'rbf':
            gamma = float(rng.choice(gammas))
            Ktr = rbf_kernel(X1, gamma=gamma)
            if X_test is not None:
                X2  = X_test[:, col_idx]
                Kte = rbf_kernel(X2, X1, gamma=gamma)

        else:  # laplacian
            gamma = float(rng.choice(gammas))
            Ktr = laplacian_kernel(X1, gamma=gamma)
            if X_test is not None:
                X2  = X_test[:, col_idx]
                Kte = laplacian_kernel(X2, X1, gamma=gamma)

        KL_train.append(Ktr)
        if X_test is not None:
            KL_test.append(Kte)

    KL_train = np.array(KL_train)
    if X_test is not None:
        return KL_train, np.array(KL_test)
    return KL_train
