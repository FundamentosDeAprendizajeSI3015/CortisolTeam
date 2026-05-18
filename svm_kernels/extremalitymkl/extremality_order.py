import numpy as np

def gram_schmidt(A):
    """
    Performs the Gram-Schmidt orthogonalization process on a matrix A.
    
    Parameters:
        A (ndarray): Input matrix of shape (m, n).
    
    Returns:
        tuple: (Q, R), where Q is an orthogonal matrix and R is an upper triangular matrix.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def matrix_rotation(u):
    """
    Performs feature matrix rotation using the Gram-Schmidt process.
    
    Parameters:
        u (ndarray): Input vector for rotation.
    
    Returns:
        ndarray: Rotation matrix.
    """
    n = len(u)
    x = np.identity(n)
    M_u = np.multiply(np.reshape(np.sign(u), (len(u), 1)), x)
    M_u[:, 0] = u / np.linalg.norm(u, 2)
    x[:, 0] = np.ones(n) / np.sqrt(n)
    M_e = x
    
    # Applying Gram-Schmidt to both matrices
    qu, ru = gram_schmidt(M_u)
    qe, re = gram_schmidt(M_e)
    
    # Computing the rotation matrix
    R_u = np.matmul(qe, np.transpose(qu))
    return R_u

def order_compar(kernels_metrics, u):
    """
    Compares kernel metrics using a rotation transformation and returns an extremality order measure.
    
    Parameters:
        kernels_metrics (ndarray): Matrix of kernel metrics.
        u (ndarray): Vector used for rotation.
    
    Returns:
        ndarray: A ranking order of kernels based on extremality.
    """
    num_kernels = np.shape(kernels_metrics)[0]
    
    # Perform rotation on the metrics
    Ru = matrix_rotation(u)
    
    # Compute new metrics after rotation
    new_metrics = np.transpose(np.dot(Ru, np.transpose(kernels_metrics)))
    
    extremality_order = np.zeros(num_kernels)
    for i in range(num_kernels):
        # Deep comparison of metrics
        extremality_order[i] = np.sum(np.all(new_metrics >= new_metrics[i], axis=1))
    
    return extremality_order
