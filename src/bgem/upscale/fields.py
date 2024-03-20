"""
Functions for construction of structured media fields.
"""
import numpy as np

voigt_coords = {
            1: [(0, 0)],
            2: [(0, 0), (1, 1), (0, 1)],
            3: [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        }

def tn_to_voigt(tn):
    """
    (dim, dim, N) -> (n_voight, N)
    :param array:
    :return:
    """
    m, n, N = tn.shape
    assert m == n
    dim = m
    tn_voigt = [
        tn[i, j, :]
        for i, j in voigt_coords[dim]
    ]
    return np.array(tn_voigt)

def _tn(k, dim):
    if type(k) == float:
        k = k * np.eye(dim)
    assert k.shape == (dim, dim)
    return k
def K_structured(points, K0, Kx=None, fx=2.0, Ky=None, fy=4.0, Q=None):
    x = points  # shape: (N, dim)
    _, dim = x.shape
    K0 = _tn(K0, dim)
    if Kx is None:
        Kx = K0
    if Ky is None:
        Ky = Kx
    Kx = _tn(Kx, dim)
    Ky = _tn(Ky, dim)
    t = 0.5 * (np.sin(2 * np.pi * fx * x[0]) + 1)
    s = 0.5 * (np.sin(2 * np.pi * fy * x[1]) + 1)
    K =  t * K0  + (1 - t) * (s * Kx + (1 - s) * Ky)
    if Q is not None:
        K = Q.T @ K @ Q
    return tn_to_voigt(K)