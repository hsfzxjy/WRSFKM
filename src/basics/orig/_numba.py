#!/usr/bin/env python3

from math import sqrt

import numpy as np
from numpy.linalg import norm as l21_norm


from numba import njit
from numba.pycc import CC

cc = CC('_numba')
cc.verbose = True


@cc.export('clip', 'f8[:](f8[:])')
@njit
def clip(x):

    w = np.zeros(x.shape)

    for i in range(x.shape[0]):
        w[i] = max(x[i].item(), 0)

    return w


@cc.export('f', 'f8(f8, f8[:])')
@njit
def f(x, u):

    n = u.shape[0]

    return clip(np.full(u.shape, x, dtype=np.float64) - u).sum() / n - x


@cc.export('df', 'f8(f8, f8[:])')
@njit
def df(x, u):
    n = u.shape[0]

    # return (x > u).sum() / n - 1
    s = 0.
    for i in range(n):
        if x > u[i].item():
            s += 1

    return s / n - 1


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
@njit(fastmath=True)
def solve_huang_eq_13(v):
    """
    min || alpha - v ||^2, subject to \sum{alpha}=1, alpha >= 0
    """

    n = v.shape[0]

    # Acceleration for:
    # u = v - np.ones((n, n)) @ v / (n) + np.ones(n) / (n)
    term2 = np.empty((n,), dtype=np.float64)
    v_sum = 0.
    for i in range(n):
        v_sum += v[i]

    for i in range(n):
        term2[i] = (v_sum - 1) / n

    u = v - term2
    # End of Acceleration

    lambda_bar_star = solve_huang_eq_24(u)
    lambda_star = (lambda_bar_star - u) + clip(u - lambda_bar_star)
    return u + lambda_star - lambda_bar_star * np.ones(n)


@cc.export('sqrnorm_2d', 'f8(f8[:, :])')
@njit(fastmath=True)
def sqrnorm_2d(x):

    result = 0.

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = x[i, j]
            result += a * a

    return result


@cc.export('sqrnorm', 'f8(f8[:])')
@njit(fastmath=True)
def sqrnorm(x):

    result = 0.

    for i in range(x.shape[0]):
        a = x[i]
        result += a * a

    return result


@cc.export('sqrdistance', 'f8(f8[:], f8[:])')
@njit(fastmath=True)
def sqrdistance(x, y):

    result = 0.

    for i in range(x.shape[0]):
        delta = x[i] - y[i]
        result += delta * delta

    return result


@cc.export('distance', 'f8(f8[:], f8[:])')
@njit(fastmath=True)
def distance(x, y):

    result = 0.

    for i in range(x.shape[0]):
        delta = x[i] - y[i]
        result += delta * delta

    return sqrt(result)


@cc.export('solve_U', '(f8[:, :], f8[:, :], f8[:, :], f8)')
@njit
def solve_U(s, x, v, gamma):

    n, ndim = s.shape
    C = len(v)

    U = np.zeros((n, C))

    for i in range(n):

        h = np.empty(shape=(C, ), dtype=np.float64)

        for j in range(C):
            h[j] = s[i, j] * sqrdistance(x[i, :], v[j, :])

        h = (-h) / (2 * gamma)
        U[i, :] = solve_huang_eq_13(h)

    return U


@cc.export('update_V', '(f8[:, :], f8[:, :], f8[:, :])')
@njit
def update_V(s, u, x):

    _, C = s.shape
    N, ndim = x.shape

    v = np.zeros((C, ndim), dtype=np.float64)

    su = (s * u).transpose()

    for k in range(C):

        sum_ = 0.
        vec_ = np.zeros((ndim, ), dtype=np.float64)

        for i in range(N):
            w = su[k, i]
            vec_ = vec_ + w * x[i, :]
            sum_ = sum_ + w

        v[k, :] = vec_ / sum_

    return v


@cc.export('update_S', '(f8[:, :], f8[:, :], f8, b1)')
@njit
def update_S(x, v, epsilon, capped):

    N = len(x)
    C = len(v)

    s = np.zeros((N, C), dtype=np.float64)

    if not capped:
        for i in range(N):
            for k in range(C):
                norm_ = distance(x[i, :], v[k, :])
                s[i, k] = norm_
    else:
        for i in range(N):
            for k in range(C):
                norm_ = distance(x[i, :], v[k, :])
                if norm_ < epsilon:
                    s[i, k] = 1 / (2 * norm_)
                # else:
                #     # print('fuck')
                #     s[i, k] = 0

    return s


@cc.export('E', '(f8[:,:], f8[:,:], f8[:,:], f8, f8, b1)')
@njit
def E(U, V, X, gamma, epsilon, capped):

    N, C, ndim = len(X), len(V), len(X[0])  # noqa

    term1 = 0.

    if capped:

        for i in range(N):
            xi = X[i, :]
            for k in range(C):
                term1 += U[i, k] * min(epsilon, distance(xi, V[k, :]))

    else:

        for i in range(N):
            xi = X[i, :]
            for k in range(C):
                term1 += U[i, k] * distance(xi, V[k, :])

    return term1 + gamma * l21_norm(U)**2


if __name__ == '__main__':
    cc.compile()
