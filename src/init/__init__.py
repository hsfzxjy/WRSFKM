import numpy as np
from utils.math_utils import l21_norm
from sklearn.cluster.k_means_ import _init_centroids


def init_uv(X, C, p):

    N, ndim = len(X), len(X[0])

    # np.random.seed()

    print(p, 'test seed', np.random.random((1,)))
    assert isinstance(p.method, str)

    if p.method == 'random':
        V = np.random.random((C, ndim))
    # elif p.method == 'orig':
    #     return origin_init(X, C, p.gamma, p.epsilon)
    elif p.method == 'kmpp':
        V = _init_centroids(X, C, 'k-means++')

    U = np.ones((N, C)) * .1 / (C - 1)

    for i in range(N):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = .9

    # w_epsilon = p.w_epsilon
    # from basics.ours import update_V as ours_update_V
    # V = ours_update_V(V, U, X, w_epsilon)

    return U, V
