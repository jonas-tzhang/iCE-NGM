import numpy as np
from tqdm import tqdm


def parabolic_3(X):
    b, e, kappa = 6, 0.1, 0.3
    g_X = b - X[:, 1] - kappa * (X[:, 0] - e) ** 2
    return g_X

def series_system(X):
    x1, x2 = X[:, 0], X[:, 1]
    val = np.zeros((X.shape[0], 4))
    val[:, 0] = 3 + (x1 - x2) ** 2 / 10 - (x1 + x2) / np.sqrt(2)
    val[:, 1] = 3 + (x1 - x2) ** 2 / 10 + (x1 + x2) / np.sqrt(2)
    val[:, 2] = x1 - x2 + 7 / np.sqrt(2)
    val[:, 3] = x2 - x1 + 7 / np.sqrt(2)

    return np.min(val, axis=1)

def four_failure_domain(X):
    x1, x2 = X[:, 0], X[:, 1]
    g_X = 15 - np.abs(x1*x2)

    return g_X