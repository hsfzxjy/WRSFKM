import numpy as np
from utils.math_utils import l21_norm


def init_uv(X, C, p):

    N, ndim = len(X), len(X[0])

    np.random.seed()

    print(p, 'test seed', np.random.random((1,)))
    assert isinstance(p.method, str)

    if p.method == 'random':
        V = np.random.random((C, ndim))
    # elif p.method == 'orig':
    #     return origin_init(X, C, p.gamma, p.epsilon)
    # else:
    #     V = _init_centroids(X, C, p.method)

    U = np.ones((N, C)) * .1 / (C - 1)

    for i in range(N):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = .9

    return U, V
