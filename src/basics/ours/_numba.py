#!/usr/bin/env python3

from math import exp

import numpy as np

import numba
from numba.pycc import CC
from numba import njit

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


@cc.export('welsch_func', 'f8[:](f8[:], f8)')
@njit
def welsch_func(x, epsilon):

    result = (1 - np.exp(- epsilon * x ** 2)) / epsilon

    return result


@cc.export('sqr_welsch_func_scalar', 'f8(f8, f8)')
@njit(fastmath=True)
def sqr_welsch_func_scalar(x, epsilon):

    result = (1 - exp(- epsilon * x)) / epsilon

    return result


@cc.export('sqr_welsch_func', 'f8[:](f8[:], f8)')
@njit(fastmath=True)
def sqr_welsch_func(x, epsilon):

    result = (1 - np.exp(- epsilon * x)) / epsilon

    return result


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


@cc.export('solve_U', 'f8[:, :](f8[:,:], f8[:,:], f8[:,:], f8, f8)')
@njit(fastmath=True, parrallel=True)
def solve_U(x, v, v_old, gamma, epsilon):

    N, C = len(x), len(v)

    # U = pymp.shared.array((N, C))

    U = np.empty(shape=(N, C), dtype=np.float64)

    for i in numba.prange(N):

        h = np.empty(shape=(C,), dtype=np.float64)
        xi = x[i, :]
        for j in range(C):
            xi_voldj_distance_sqr = sqrdistance(xi, v_old[j, :])
            xi_vj_distance_sqr = sqrdistance(xi, v[j, :])
            W_j = sqr_welsch_func_scalar(xi_voldj_distance_sqr, epsilon)
            h[j] = W_j + (np.sqrt(xi_vj_distance_sqr) - np.sqrt(xi_voldj_distance_sqr)) * (1 - epsilon * W_j)
            h[j] = -h[j] / (2 * gamma)

        U[i, :] = solve_huang_eq_13(h)

    return U


@cc.export('update_V', 'f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8)')
@njit(fastmath=True, parrallel=True)
def update_V(v, u, x, epsilon):

    N, C, ndim = len(x), len(v), len(x[0])  # noqa

    new_v = np.zeros(v.shape, dtype=np.float64)

    # Change memory access order to accelerate
    xt = x.transpose().copy()
    ut = u.transpose().copy()

    for k in range(C):
        vec = np.empty((N,), dtype=np.float64)

        denominator = 0.
        vk = v[k, :]

        for i in range(N):
            tmp = vec[i] = ut[k, i] * exp(
                -epsilon *
                sqrdistance(x[i, :], vk)
            )

            denominator += tmp

        # Acceleration for:
        # new_v[k, :] = vec @ x / denominator
        for j in range(ndim):

            tmp = 0.
            for i in range(N):
                tmp += vec[i] * xt[j, i]
            new_v[k, j] = tmp / denominator

    return new_v


@cc.export('E', 'f8(f8[:,:],f8[:,:],f8[:,:],f8,f8)')
@njit(parrallel=True)
def E(U, V, X, gamma, epsilon):

    N, C, ndim = len(X), len(V), len(X[0])  # noqa

    term1 = 0.
    for i in range(N):
        xi = X[i, :]
        for k in range(C):
            term1 += U[i, k] * sqr_welsch_func_scalar(sqrdistance(xi, V[k, :]), epsilon)

    return term1 + gamma * sqrnorm_2d(U)


if __name__ == '__main__':
    cc.compile()
