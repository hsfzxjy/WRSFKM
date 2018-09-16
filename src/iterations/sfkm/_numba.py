#!/usr/bin/env python3

import numpy as np
from numba.pycc import CC
from numba import njit
from math import sqrt
import numba

cc = CC('_numba')


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

#


@cc.export('solve_U', 'f8[:, :](f8[:,:], f8[:,:], f8)')
@njit(fastmath=True, parrallel=True)
def solve_U(x, v, gamma):

    N, C = len(x), len(v)

    # U = pymp.shared.array((N, C))

    U = np.empty(shape=(N, C), dtype=np.float64)

    for i in numba.prange(N):

        h = np.empty(shape=(C,), dtype=np.float64)
        xi = x[i, :]
        for j in range(C):
            h[j] = - (sqrdistance(xi, v[j, :])) / (2 * gamma)

        # h = - w / (2 * gamma)
        # h = - sqr_welsch_func(w, epsilon) / (2 * gamma)
        U[i, :] = solve_huang_eq_13(h)

    return U


@cc.export('distance', 'f8(f8[:], f8[:])')
@njit(fastmath=True)
def distance(x, y):

    result = 0.

    for i in range(x.shape[0]):
        delta = x[i] - y[i]
        result += delta * delta

    return sqrt(result)


@cc.export('sqrdistance', 'f8(f8[:], f8[:])')
@njit(fastmath=True)
def sqrdistance(x, y):

    result = 0.

    for i in range(x.shape[0]):
        delta = x[i] - y[i]
        result += delta * delta

    return result


@cc.export('update_V', '(f8[:,:], f8[:,:], f8[:,:], f8)')
@njit
def update_V(X, U, V, gamma):
    N, ndim = X.shape
    C = len(V)
    for j in range(C):
        fenzi = np.zeros((ndim,), dtype=np.float64)
        fenmu = 0
        for i in range(N):
            fenzi = fenzi + (U[i, j]) * X[i, :]
            fenmu = fenmu + (U[i, j])
        V[j, :] = fenzi / fenmu
    return V


@cc.export('E', '(f8[:,:],f8[:,:],f8[:,:], f8)')
@njit
def E(X, U, V, gamma):
    N = len(X)
    C = len(V)
    E = 0
    for i in range(N):
        for k in range(C):
            E = E + (U[i, k]) * sqrdistance(X[i, :], V[k, :]) + gamma * U[i, k] * U[i, k]
    return E


if __name__ == '__main__':
    cc.compile()
