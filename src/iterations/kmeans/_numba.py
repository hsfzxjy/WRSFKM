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


@cc.export('update_U', '(f8[:,:], f8[:,:], f8[:,:])')
@njit
def update_U(X, V, U):
    N = len(X)
    C = len(V)
    U = np.zeros((N, C), dtype=np.float64)

    for i in range(N):
        dist = np.zeros((C,), dtype=np.float64)
        for k in range(C):
            dist[k] = distance(X[i, :], V[k, :])
        U[i, np.argmin(dist)] = 1
    return U


@cc.export('update_V', '(f8[:,:], f8[:,:], f8[:,:])')
@njit
def update_V(X, U, V):
    N, ndim = X.shape
    C = len(V)
    for i in range(C):
        count = 0
        sum = np.zeros((ndim, ), dtype=np.float64)
        for k in range(N):
            if U[k, i] == 1:
                count = count + 1
                sum = sum + X[k, :]
        if count != 0:
            V[i, :] = sum / count

    return V


@cc.export('E', '(f8[:,:], f8[:,:], f8[:,:])')
@njit
def E(X, U, V):
    N = len(X)
    C = len(V)
    E = 0
    for i in range(N):
        for k in range(C):
            if U[i, k] == 1:
                E = E + sqrdistance(X[i, :], V[k, :])
    return E


if __name__ == '__main__':
    cc.compile()
