import numpy as np

def complex_ratio(A):
    """Computes the trace of a matrix."""
    return np.trace(A)

def FSM(K, y):
    """Computes the Feature Space Measure (FSM)."""
    n_nega = np.count_nonzero(y == -1)  # Number of negative class samples
    n_posi = len(y) - n_nega  # Number of positive class samples

    # Compute intra-class and inter-class similarities
    d_i = np.sum(K[np.ix_(y == -1, y == -1)], axis=1) / n_nega  # Negative class intra-similarity
    a_i = np.sum(K[np.ix_(y == 1, y == 1)], axis=1) / n_posi   # Positive class intra-similarity
    c_i = np.sum(K[np.ix_(y == -1, y == 1)], axis=1) / n_posi  # Negative-positive inter-similarity
    b_i = np.sum(K[np.ix_(y == 1, y == -1)], axis=1) / n_nega  # Positive-negative inter-similarity

    # Compute global similarity measures
    A = sum(a_i) / n_posi
    B = sum(b_i) / n_posi
    C = sum(c_i) / n_nega
    D = sum(d_i) / n_nega

    rest_phi_square = A + D - B - C  # Normalization factor
    aux_1 = np.true_divide(sum(np.square(b_i - a_i + A - B)), rest_phi_square * (n_posi - 1))
    aux_2 = np.true_divide(sum(np.square(c_i - d_i + D - C)), rest_phi_square * (n_nega - 1))

    return np.true_divide(np.sqrt(aux_1) + np.sqrt(aux_2), np.sqrt(rest_phi_square))

def ideal_kernel(y):
    """Constructs the ideal kernel based on labels."""
    K_ideal = np.equal.outer(y, y).astype(int)  # Creates a similarity matrix
    K_ideal = np.where(K_ideal == 0, -1 + K_ideal, K_ideal)  # Assigns -1 to different class pairs
    return K_ideal

def kernel_aligment(K, y):
    """Computes the kernel alignment."""
    A1 = np.trace(np.dot(K.transpose(), ideal_kernel(y)))  # Inner product with ideal kernel
    return A1 / (np.linalg.norm(K, 'fro') * len(y))  # Normalized alignment score

def kernel_polarization(k, y):
    """Computes kernel polarization."""
    n1 = len(y)
    A = np.zeros((n1, n1))
    
    # Compute polarization between all sample pairs
    for i in range(n1):
        for j in range(i+1, n1):
            A[i, j] = -1 * y[i] * y[j] * (k[i, i] + k[j, j] - 2 * k[i, j])
            A[j, i] = A[i, j]  # Ensure symmetry

    return np.sum(np.matrix(A))  # Return total polarization measure
