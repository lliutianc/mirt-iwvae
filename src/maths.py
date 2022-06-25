import numpy as np
import scipy.stats


DELTA = 1e-8
TOLERANCE = 1e-4


def primes(n):
    odds = range(3, n + 1, 2)
    sieve = set(sum([list(range(q * q, n + 1, q + q)) for q in odds], []))
    return [2] + [p for p in odds if p not in sieve]


def sigmoid(x):
    return 1 / (1 + np.exp(-x) + DELTA)


# Fast broadcasting function.
def ext_outprod(W):
    """
    Vectorize outerproduct of each row Wj @ Wj.T.
    W should be a matrix or a vector.
    :param W: Matrix: (p, q) , Vector: both (p, 1) and (1, p) are acceptable.
    :return: WOUT: 3D matrix: (p, q, q)
    """
    if len(W.shape) == 1:
        W = W.reshape(1, W.shape[0])
    WOUT = np.expand_dims(W.T, 2) * W
    return WOUT.transpose(1, 0, 2)


def is_converge(v1, v2, accuracy=TOLERANCE):
    return np.abs(v1 - v2).max() < accuracy


def proj(X):
    return X @ np.linalg.pinv(X.T @ X) @ X.T