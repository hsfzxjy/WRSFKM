#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm as l21_norm


from numba import njit
from numba.pycc import CC

cc = CC('orig_accelerate')
cc.verbose = True


@cc.export('clip', 'f8[:](f8[:])')
@njit
def clip(x):

    w = np.zeros(x.shape)

    for i in range(x.shape[0]):
        item = x[i].item()

        if item > 0:
            w[i] = item

    return w


@cc.export('f', 'f8(f8, f8[:])')
@njit
def f(x, u):

    n = len(u)

    return clip(np.full(u.shape, x, dtype=np.float64) - u).sum() / n - x


@cc.export('df', 'f8(f8, f8[:])')
@njit
def df(x, u):
    n = len(u)

    return (x > u).sum() / n - 1


@cc.export('solve_huang_eq_24', 'f8(f8[:,:])')
@njit
def solve_huang_eq_24(u):

    EPS = 1e-4

    lamb = np.min(u)
    while True:
        new_lamb = lamb - f(lamb, u) / df(lamb, u)
        if np.abs(new_lamb - lamb) < EPS:
            return new_lamb

        lamb = new_lamb


@cc.export('solve_huang_eq_13', 'f8[:](f8[:])')
@njit
def solve_huang_eq_13(v):
    """
    min || alpha - v ||^2, subject to \sum{alpha}=1, alpha >= 0
    """

    n = len(v)
    u = v - np.ones((n, n)) @ v / (n) + np.ones(n) / (n)
    lambda_bar_star = solve_huang_eq_24(u)
    lambda_star = (lambda_bar_star - u) + clip(u - lambda_bar_star)
    return u + lambda_star - lambda_bar_star * np.ones(n)


@cc.export('solve_U', '(f8[:, :], f8[:, :], f8[:, :], f8)')
@njit
def solve_U(s, x, v, gamma):

    n, ndim = s.shape
    C = len(v)

    U = np.zeros((n, C))

    for i in range(n):

        h = np.empty(shape=(C, ), dtype=np.float64)

        for j in range(C):
            h[j] = s[i, j] * l21_norm(x[i, :] - v[j, :]) ** 2

        h = (-h) / (2 * gamma)
        U[i, :] = solve_huang_eq_13(h)

    return U


# @cc.export('update_V', '(f8[:, :], f8[:, :], f8[:, :])')
# @njit
# def update_V(s, u, x):

#     ndim, C = s.shape

#     v = np.zeros((C, ndim))

#     su = s * u

#     for k in range(C):
#         v[k, :] = np.average(x, axis=0, weights=su[:, k])

#     return v


@cc.export('update_S', '(f8[:, :], f8[:, :], f8)')
@njit
def update_S(x, v, epsilon):

    N = len(x)
    C = len(v)

    s = np.ones((N, C))

    for i in range(N):
        for k in range(C):
            norm_ = l21_norm(x[i, :] - v[k, :])
            s[i, k] = norm_
            # if norm_ < epsilon:
            #     s[i, k] = 1 / (2 * norm_)
            # else:
            #     s[i, k] = 0

    return s


@cc.export('E', '(f8[:,:],f8[:,:],f8[:,:],f8,f8)')
@njit
def E(U, V, X, gamma, epsilon):

    N, C, ndim = len(X), len(V), len(X[0])  # noqa

    W = np.empty(shape=(N, C), dtype=np.float64)
    for i in range(N):
        xi = X[i, :]
        for k in range(C):
            W[i, k] = l21_norm(xi - V[k, :])

    return np.sum(U * W) + gamma * l21_norm(U)**2


if __name__ == '__main__':
    cc.compile()
