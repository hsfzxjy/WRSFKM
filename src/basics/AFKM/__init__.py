import numpy as np
from ._numba import solve_U, E


def update_V(u, x):

    _, C = s.shape
    N, ndim = x.shape

    v = np.zeros((C, ndim))


    for k in range(C):
        v[k, :] = np.average(x, axis=0, weights=u[:, k])

    return v
