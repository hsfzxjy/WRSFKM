#!/usr/bin/env python3

import numpy as np
from numba.pycc import CC
from numba import njit
from math import sqrt

cc = CC('_numba')


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


@cc.export('update_U', '(f8[:,:], f8[:,:], f8)')
@njit
def update_U(X, V, m):
    N = len(X)
    C = len(V)
    U = np.zeros((N, C), dtype=np.float64)
    exponent = 2 / (m - 1)

    dist = np.zeros((N, C), dtype=np.float64)
    for i in range(N):
        for j in range(C):
            dist[i, j] = distance(X[i, :], V[j, :]) ** exponent
            print(dist[i, j], distance(X[i, :], V[j, :]))

    for i in range(N):

        s = 0.
        for k in range(C):
            s = s + 1 / dist[i, k]

        print(s)
        for j in range(C):
            U[i, j] = 1 / (dist[i, j] * s)

    return U


@cc.export('update_V', '(f8[:,:], f8[:,:], f8[:,:], f8)')
@njit
def update_V(X, U, V, m):
    N = len(X)
    C, ndim = V.shape
    new_V = np.zeros(V.shape, dtype=np.float64)

    Ut = (U.T).copy()

    for j in range(C):
        fenzi = np.zeros((ndim,), dtype=np.float64)
        fenmu = 0.
        for i in range(N):
            Uij_pow_m = Ut[j, i] ** m
            fenzi = fenzi + Uij_pow_m * X[i, :]
            fenmu = fenmu + Uij_pow_m
        new_V[j, :] = fenzi / fenmu
    return new_V


@cc.export('E', '(f8[:,:],f8[:,:],f8[:,:], f8)')
@njit
def E(X, U, V, m):
    N = len(X)
    C = len(V)
    E = 0
    for i in range(N):
        for k in range(C):
            E = E + U[i, k] ** m * sqrdistance(X[i, :], V[k, :])
    return E


if __name__ == '__main__':
    cc.compile()
