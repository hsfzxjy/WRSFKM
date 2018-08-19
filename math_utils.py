#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm as l21_norm

from numba.pycc import CC
from numba import njit

cc = CC('math_utils')
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


@cc.export('welsch_func', 'f8[:](f8[:], f8)')
@njit
def welsch_func(x, epsilon):

    result = (1 - np.exp(- epsilon * x ** 2)) / epsilon

    return result


@cc.export('solve_U', 'f8[:, :](f8[:,:], f8[:,:], f8, f8)')
def solve_U(x, v, gamma, epsilon):

    N, C = len(x), len(v)

    # U = pymp.shared.array((N, C))

    U = np.empty(shape=(N, C), dtype=np.float64)

    for i in range(N):

        w = np.empty(shape=(C,), dtype=np.float64)
        for j in range(C):
            w[j] = l21_norm(x[i, :] - v[j, :])

        h = welsch_func(w, epsilon)
        h = (-h) / (2 * gamma)
        U[i, :] = solve_huang_eq_13(h)

    return U


@cc.export('update_V', 'f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8)')
def update_V(v, u, x, epsilon):

    N, C, ndim = len(x), len(v), len(x[0])  # noqa

    W = np.empty((C, N), dtype=np.float64)

    for k in range(C):
        for i in range(N):
            W[k, i] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)

    new_v = np.zeros(v.shape)

    for k in range(C):
        denominator = W[k, :].sum()

        new_v[k, :] = W[k, :].reshape((1, N)) @ x / denominator

    return new_v


@cc.export('origin_solve_U', 'f8[:,:](f8[:,:],f8[:,:],f8,f8)')
@njit
def origin_solve_U(x, v, gamma, epsilon):

    N, C, ndim = len(x), len(v), len(x[0])  # noqa

    U = np.zeros((N, C))
    for i in range(N):
        # xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
        h = np.empty(shape=(C,), dtype=np.float64)
        for j in range(C):
            h[j] = l21_norm(x[i, :] - v[j, :])
        # h = l21_norm(xi - v, axis=1)
        h = (-h) / (4 * gamma * epsilon**0.5 / 0.63817562)
        U[i, :] = solve_huang_eq_13(h)
    return U


@cc.export('origin_update_V', 'f8[:,:](f8[:,:],f8[:,:],f8[:,:])')
@njit
def origin_update_V(x, u, v):

    N, C, ndim = len(x), len(v), len(x[0])  # noqa

    V = np.zeros((C, ndim))
    for k in range(C):
        A = 0
        vk = v[k, :]  # .reshape((1, ndim))
        for i in range(N):
            xi = x[i, :]  # .reshape((1, ndim))
            V[k, :] = V[k, :] + 1 / (2 * l21_norm(xi - vk)) * u[i, k] * xi
            A = A + 1 / (2 * l21_norm(xi - vk)) * u[i, k]
        V[k, :] = V[k, :] / A
    return V


@cc.export('origin_init', '(f8[:,:], i8, f8, f8)')
def origin_init(X, C, gamma, epsilon):

    N, ndim = len(X), len(X[0])
    size = N

    V = np.empty((C, ndim), dtype=np.float64)

    for i in range(C):
        for k in range(ndim):
            V[i, k] = np.random.random()

    U = np.zeros((size, C))
    t = 0
    while True:
        U = origin_solve_U(X, V, gamma, epsilon)
        new_V = origin_update_V(X, U, V)
        delta = l21_norm(new_V - V)
        V = new_V
        print('init t =', t)
        if delta < 1e-1:
            break
        t += 1

    return U, V


@cc.export('E', 'f8(f8[:,:],f8[:,:],f8[:,:],f8,f8)')
def E(U, V, X, gamma, epsilon):

    N, C, ndim = len(X), len(V), len(X[0])

    W = np.empty(shape=(N, C), dtype=np.float64)
    for i in range(N):
        xi = X[i, :]
        for k in range(C):
            W[i, k] = welsch_func(l21_norm(xi - V[k, :]), epsilon)

    return np.sum(U * W) + gamma * l21_norm(U)**2


if __name__ == '__main__':
    cc.compile()
