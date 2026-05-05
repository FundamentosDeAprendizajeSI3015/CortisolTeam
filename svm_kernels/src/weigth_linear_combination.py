import numpy as np

def weight(b, n):
    x = np.divide(b, np.sum(b))
    x = np.power(x, n)
    x = np.divide(x, np.sum(x))
    return x

