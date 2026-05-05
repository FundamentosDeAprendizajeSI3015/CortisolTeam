"""
Weak polynomial kernel generator.

Adapted from https://github.com/maospina1041/extremality_mkl
Applied to the Crypto ML Project (CortisolTeam).

A "weak kernel" is built from a random subset of the available features
(technical indicators such as RSI, MACD, SMA, ATR, etc.) raised to a
random polynomial degree.  Combining many weak kernels via extremality
weights forms a strong, data-adaptive kernel for SVM classification.
"""

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel


def create_weak_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
    num_kernels: int = 20,
    max_features: int = 8,
    max_degree: int = 3,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
    """Generate a set of weak polynomial kernels by random feature sub-sampling.

    Each weak kernel is computed on a random subset of features (with
    replacement) and uses a random polynomial degree in [1, max_degree].
    The diversity introduced by random sub-sampling allows the extremality
    weighting step to select complementary kernels that together capture
    different aspects of the crypto feature space.

    Args:
        X_train: Training feature matrix of shape (n_train, n_features).
                 Features are the technical indicators (RSI, MACD, SMA …).
        X_test:  Test feature matrix of shape (n_test, n_features).
                 When provided, a matching set of test kernels is returned.
        num_kernels:  Number of weak kernels to generate.
        max_features: Maximum number of features selected per kernel.
                      Should be ≤ X_train.shape[1].
        max_degree:   Maximum polynomial degree (≥ 1).
        random_state: Seed for reproducibility.

    Returns:
        If X_test is None:
            KL_train — array of shape (num_kernels, n_train, n_train).
        If X_test is provided:
            (KL_train, KL_test) where KL_test has shape
            (num_kernels, n_test, n_train).

    Notes:
        - Feature columns are sampled *with replacement*, allowing a single
          indicator (e.g. rsi_14) to receive extra weight in some kernels.
        - polynomial_kernel uses the homogeneous form: K(x,y) = (x·y)^d.
          gamma=1 and coef0=0 keep the kernel scale consistent with the
          MinMaxScaler normalisation applied upstream.
    """
    rng = np.random.default_rng(random_state)
    n_features = X_train.shape[1]

    KL_train = []
    KL_test = [] if X_test is not None else None

    for _ in range(num_kernels):
        # Sample a random subset of feature indices (with replacement)
        k = rng.integers(1, max_features + 1)
        col_idx = rng.integers(0, n_features, size=k)

        degree = int(rng.integers(1, max_degree + 1))

        X1 = X_train[:, col_idx]
        KL_train.append(
            polynomial_kernel(X1, degree=degree, gamma=1.0, coef0=0)
        )

        if X_test is not None:
            X2 = X_test[:, col_idx]
            KL_test.append(
                polynomial_kernel(X2, X1, degree=degree, gamma=1.0, coef0=0)
            )

    KL_train = np.array(KL_train)
    if X_test is not None:
        return KL_train, np.array(KL_test)
    return KL_train
