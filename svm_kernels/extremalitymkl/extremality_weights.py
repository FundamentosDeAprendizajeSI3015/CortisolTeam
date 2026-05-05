import numpy as np
from src.kernel_metrics import complex_ratio, kernel_aligment, FSM, kernel_polarization
from src.weigth_linear_combination import weight
from .extremality_order import order_compar

# Define the available metrics
METRICS = {
    "alignment": kernel_aligment,
    "polarization": kernel_polarization,
    "FSM": FSM,
    "complex_ratio": complex_ratio
}

# Define the direction of each metric (+1 for performance, -1 for error)
DIRECTIONS = {
    "alignment": 1,
    "polarization": 1,
    "FSM": 1,
    "complex_ratio": -1
}

def metrics_kernels(KL_train, y_train, metrics=None):
    """
    Computes kernel-based metrics for a given set of kernel matrices and labels.
    
    Parameters:
        KL_train (array-like): A set of kernel matrices.
        y_train (array-like): Training labels.
        metrics (dict, optional): Dictionary of selected metrics to compute. If None, uses all available metrics.
    
    Returns:
        tuple: A matrix of computed metric values and an array of metric directions.
    """
    if metrics is None:
        metrics = METRICS  # Use all metrics by default
    
    num_kernels = np.shape(KL_train)[0]  # Number of kernels
    num_metrics = len(metrics)  # Number of selected metrics

    measures = np.zeros((num_kernels, num_metrics))  # Matrix to store metric values

    for i in range(num_kernels):
        measures[i] = [metrics[m](KL_train[i], y_train) for m in metrics]

    directions = np.array([DIRECTIONS[m] for m in metrics])  # Array with metric directions

    return measures, directions

# Custom set of metrics used for kernel weighting
custom_metrics = { "alignment": kernel_aligment, "FSM": FSM }

class KernelWeights:
    """
    Stores weights for two different kernel weighting strategies.
    
    Attributes:
        w_1 (array-like): Weights for the first extremality order.
        w_2 (array-like): Weights for the second extremality order.
    """
    def __init__(self, w_1, w_2):
        self.w_1 = w_1
        self.w_2 = w_2

def kernel_extremaly_weights(KL_train, y_train, metrics=custom_metrics, n=1):
    """
    Computes kernel weights using an extremality-based ordering approach.
    
    Parameters:
        KL_train (array-like): A set of kernel matrices.
        y_train (array-like): Training labels.
        metrics (dict, optional): Dictionary of selected metrics for ordering. Default is custom_metrics.
        n (int, optional): Parameter for weight calculation. Default is 1.
    
    Returns:
        KernelWeights: An object containing two sets of kernel weights.
    """
    metrics_KL_train, direction = metrics_kernels(KL_train, y_train, metrics)
    
    # Compute weights for the first extremality order
    order_extremality_kernel_1 = order_compar(metrics_KL_train, direction)
    w_1 = weight(len(order_extremality_kernel_1) - order_extremality_kernel_1, n)

    # Compute weights for the second extremality order
    order_extremality_kernel_2 = order_compar(metrics_KL_train, -1 * direction)
    w_2 = weight(order_extremality_kernel_2, n)

    return KernelWeights(w_1, w_2)
