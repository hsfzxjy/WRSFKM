#!/usr/bin/env python3

import numpy as np
from numba.pycc import CC
from numba import njit
from math import sqrt, exp, log


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
def update_U(X, V, gamma):
    N = len(X)
    C = len(V)
    U = np.zeros((N, C), dtype=np.float64)
    for i in range(N):

        expdist = np.empty((C, ), dtype=np.float64)

        for k in range(C):
            expdist[k] = exp(-distance(X[i, :], V[k, :]) / gamma)

        s = expdist.sum()
        U[i, :] = expdist / s

    return U


@cc.export('update_V', '(f8[:,:], f8[:,:], f8[:,:], f8)')
@njit
def update_V(X, U, V, gamma):
    N, ndim = X.shape
    C = len(V)
    V = np.zeros(V.shape, dtype=np.float64)
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
            E = E + (U[i, k]) * distance(X[i, :], V[k, :]) + gamma * U[i, k] * log(U[i, k])
    return E


if __name__ == '__main__':
    cc.compile()
