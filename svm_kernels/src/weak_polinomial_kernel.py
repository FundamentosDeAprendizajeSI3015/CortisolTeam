from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np

def create_weak_kernels(X_train, X_test=None, t=5, num_kernels=3, max_degree=3):
    """
    Generates a set of weak polynomial kernels by randomly selecting feature subsets 
    and applying polynomial transformations with varying degrees.
    
    Parameters:
        X_train (array-like): Training data.
        X_test (array-like, optional): Testing data. Default is None.
        t (int): Maximum number of selected features per kernel.
        num_kernels (int): Number of weak kernels to generate.
        max_degree (int): Maximum polynomial degree for the kernels.

    Returns:
        tuple or ndarray: If X_test is provided, returns (KL_train, KL_test), otherwise just KL_train.
    """
    KL_train = []
    KL_test = None if X_test is None else []

    for i in range(num_kernels):
        # Selection of column indices **with repetition**
        index_columns = np.random.randint(0, X_train.shape[1], size=np.random.randint(1, t + 1))

        X1 = X_train[:, index_columns]  # Allows selection with repetition
        degree = np.random.randint(1, max_degree + 1)  # Random polynomial degree

        KL_train.append(polynomial_kernel(X1, degree=degree, coef0=0, gamma=1))

        if X_test is not None:
            X2 = X_test[:, index_columns]  # Ensures the same selected columns are used
            KL_test.append(polynomial_kernel(X2, X1, degree=degree, coef0=0, gamma=1))

    KL_train = np.array(KL_train)
    KL_test = np.array(KL_test) if X_test is not None else None

    return (KL_train, KL_test) if X_test is not None else KL_train
