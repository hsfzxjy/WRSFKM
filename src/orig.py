#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm as l21_norm

from utils.metrics import metric
from utils.math_utils import U_converged

from orig_accelerate import solve_U, update_S, E


# def update_S(x, v, epsilon):

#     N = len(x)
#     C = len(v)

#     s = np.ones((N, C))

#     for i in range(N):
#         for k in range(C):
#             norm_ = l21_norm(x[i, :] - v[k, :])
#             s[i, k] = 1 / (2 * norm_)
#             # if norm_ < epsilon:
#             #     s[i, k] = 1 / (2 * norm_)
#             # else:
#             #     s[i, k] = 0

#     return s


# def solve_U(s, x, v, gamma):

#     n, ndim = x.shape
#     C = len(v)

#     U = np.zeros((n, C))

#     for i in range(n):
#         xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
#         h = s[i, :] * l21_norm(xi - v, axis=1) ** 2
#         h = (-h) / (2 * gamma)
#         U[i, :] = solve_huang_eq_13(h)

#     return U


def update_V(s, u, x):

    _, C = s.shape
    N, ndim = x.shape

    v = np.zeros((C, ndim))

    su = s * u

    for k in range(C):
        v[k, :] = np.average(x, axis=0, weights=su[:, k])

    return v


def orig_iteration(X, U, V, labels, p, logger):

    N = len(X)
    C = len(V)

    gamma, epsilon = p.gamma, p.epsilon

    S = np.ones((N, C))
    t = 0
    while True:
        new_U = solve_U(S, X, V, gamma)
        delta, converged = U_converged(new_U, U)
        print(delta)
        U = new_U
        V = update_V(S, U, X)
        S = update_S(X, V, epsilon)
        metric_now = metric(U, labels)
        E_now = E(U, V, X, gamma, epsilon)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now


if __name__ == '__main__':

    from tester import DualTester

    # DualTester(
    #     root_directory='',
    #     init_params=dict(method='random'),
    #     mutual={'epsilon': 30, 'gamma': .1},
    #     dataset='mnist_10k',
    #     params={
    #         'orig_random': {'iter_method': 'orig'},
    #     },
    #     times=1
    # ).execute()

    DualTester(
        root_directory='',
        init_params=dict(method='random'),
        mutual={'epsilon': 30, 'gamma': .05},
        dataset='coil_20',
        params={
            'orig_random': {'iter_method': 'orig'},
        },
        times=1
    ).execute()
