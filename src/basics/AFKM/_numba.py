from math import exp
import math

import numpy as np
from numpy.linalg import norm as l21_norm

import numba
from numba import njit
from numba.pycc import CC

def sqrdistance(x, y):

    result = 0.

    for i in range(x.shape[0]):
        delta = x[i] - y[i]
        result += delta * delta

    return result

def sqrnorm_2d(x):

    result = 0.

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = x[i, j]
            result += a * a

    return result

def solve_U(X, V, gamma):
	N, C, ndim = len(X), len(V), len(X[0])

	term2 = 0.
	for i in range(N):
        xi = x[i, :]
        for k in range(C):
            for j in range(C):
            U[i, k] +=np.exp(-sqrdistance(xi,v[j, :]) / gamma)
        U[i, k] =np.exp(-sqrdistance(xi,v[k, :]) / gamma) / U[i, k]

    return U


def E(U, V, X, gamma):

    N, C, ndim = len(X), len(V), len(X[0])  # noqa

    term1 = 0.
    for i in range(N):
        xi = X[i, :]
        for k in range(C):
            term1 += U[i, k] * sqrdistance(xi, V[k, :])
            term1 += gamma * U[i, k] * math.log(U[i,k])

    return term1 
