#!/usr/bin/env python3

"""
Requires:

python-mnist
numpy
sklearn

Anderson Acceleration
"""

import mnist
import numpy as np
import scipy
from numpy.linalg import norm as l21_norm
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.cluster.k_means_ import _init_centroids

gamma = .001
epsilon = 1e-4

# Download t10k_* from http://yann.lecun.com/exdb/mnist/
# Change to directory containing unzipped MNIST data
mndata = mnist.MNIST('./MNIST/')


def solve_huang_eq_24(u):

    n = len(u)

    def f(x):
        return np.clip(x - u, 0, None).sum() / n - x

    def df(x):
        return (x > u).sum() / n - 1

    EPS = 1e-4

    lamb = np.min(u)
    while True:
        new_lamb = lamb - f(lamb) / df(lamb)
        if np.abs(new_lamb - lamb) < EPS:
            return new_lamb

        lamb = new_lamb


def solve_huang_eq_13(v):
    """
    min || alpha - v ||^2, subject to \sum{alpha}=1, alpha >= 0
    """

    n = len(v)
    u = v - np.ones((n, n)) @ v / (n) + np.ones(n) / (n)
    lambda_bar_star = solve_huang_eq_24(u)
    lambda_star = (lambda_bar_star - u) + np.clip(u - lambda_bar_star, 0, None)
    return u + lambda_star - lambda_bar_star * np.ones(n)


def welsch_func(x):

    result = (1 - np.exp(- epsilon * x ** 2)) / epsilon

    return result


# from opti_u import solve as solve_huang_eq_13_new
import pymp
import multiprocessing as mp


def solve_U(x, v, gamma):

    N, C, ndim = len(x), len(v), len(x[0])

    U = pymp.shared.array((N, C))

    with pymp.Parallel(mp.cpu_count()) as p:
        for i in p.range(N):
            xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
            h = welsch_func(l21_norm(xi - v, axis=1))
            h = (-h) / (2 * gamma)
            U[i, :] = solve_huang_eq_13(h)

    return U


def update_V(v, u, x):

    N, C, ndim = len(x), len(v), len(x[0])  # noqa

    W = np.zeros((N, C))

    for i in range(N):
        for k in range(C):
            W[i, k] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)

    new_v = np.zeros(v.shape)

    for k in range(C):
        denominator = W[:, k].sum()

        new_v[k, :] = W[:, k].reshape((1, N)) @ x / denominator

    return new_v


def NMI(U, labels):

    return nmi(labels, np.argmax(U, axis=1))


def origin_init(X, C):

    N, ndim = len(X), len(X[0])
    size = N

    def origin_solve_U(x, v, gamma):
        U = np.zeros((N, C))
        for i in range(N):
            xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
            h = l21_norm(xi - v, axis=1)
            h = (-h) / (4 * gamma * epsilon**0.5 / 0.63817562)
            U[i, :] = solve_huang_eq_13(h)
        return U

    def origin_update_V(x, u, v):
        V = np.zeros((C, ndim))
        for k in range(C):
            A = 0
            vk = v[k, :].reshape((1, ndim))
            for i in range(N):
                xi = x[i, :].reshape((1, ndim))
                V[k, :] = V[k, :] + 1 / (2 * l21_norm(xi - vk)) * u[i, k] * xi
                A = A + 1 / (2 * l21_norm(xi - vk)) * u[i, k]
            V[k, :] = V[k, :] / A
        return V

    V = np.random.random((C, ndim))
    U = np.zeros((size, C))
    while True:
        U = origin_solve_U(X, V, gamma)
        new_V = origin_update_V(X, U, V)
        delta = l21_norm(new_V - V)
        V = new_V
        print(delta)
        if delta < 1e-1:
            break

    return U, V


from math_utils import solve_U, update_V, origin_init  # noqa


def init_uv(X, C, *, method, gamma=gamma, epsilon=epsilon):

    N, ndim = len(X), len(X[0])

    assert isinstance(method, str)

    if method == 'random':
        V = np.random.random((C, ndim))
    elif method == 'orig':
        return origin_init(X, C, gamma, epsilon)
    else:
        V = _init_centroids(X, C, method)

    U = np.ones((N, C)) * .1 / (C - 1)

    for i in range(N):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = .9

    return U, V


def run(X, C, labels, *, logger=None, init=None, iter_method=None, initial=None):

    assert isinstance(iter_method, str)

    if init == 'preset':
        U, V = initial
    else:
        U, V = init_uv(X, C, method=init)

    if iter_method.lower() in ['sv', 'mv']:
        return normal_iteration(X, C, U, V, labels, logger=logger, multi_V=iter_method.lower() == 'mv')
    elif iter_method.lower() == 'aa':
        return anderson_iteration(X, C, U, V, labels, logger=logger)


def U_converged(old, new, tol=1e-1):

    delta = l21_norm(old - new)

    return delta, delta < tol


from scipy.linalg import qr_delete


from math_utils import E


def anderson_iteration(X, C, U, V, labels, *, logger=None):

    log = logger.print if logger else __builtins__.print

    V_len = V.flatten().shape[0]
    mAA = 0

    V_old = V
    U_old = U

    iterations = t = 0
    mmax = 4
    AAstart = 0
    droptol = 1e10

    fold = gold = None
    g = np.ndarray(shape=(V_len, 0))
    Q = np.ndarray(shape=(V_len, 1))
    R = np.ndarray(shape=(1, 1))

    old_E = np.Infinity
    VAUt = None

    while True:
        log('t =', t)
        U_new = solve_U(X, V_old, gamma, epsilon)

        delta_U, is_converged = U_converged(U_new, U_old)
        log('DELTA U', delta_U)

        new_E = E(U_new, V_old, X, gamma, epsilon)

        if is_converged:
            return t, NMI(U_new, labels)

        if new_E >= old_E:
            # reset
            log('reset')
            mAA = 0
            iterations = 0
            V_old = VAUt
            g = np.ndarray(shape=(V_len, 0))
            Q = np.ndarray(shape=(V_len, 1))
            R = np.ndarray(shape=(1, 1))
            U_new = solve_U(X, V_old, gamma, epsilon)
            old_E = E(U_new, V_old, X, gamma, epsilon)

        # AA Start
        VAUt = gcur = update_V(V_old, U_new, X, epsilon)
        # gcur = multi_update_V(V_old, U_new, X)
        fcur = gcur - V_old

        if iterations > AAstart:
            delta_f = (fcur - fold).reshape((V_len, 1))
            delta_g = (gcur - gold).reshape((V_len, 1))

            if mAA < mmax:
                g = np.hstack((g, delta_g))
            else:
                g = np.hstack((g[:, 1:mAA], delta_g))

            mAA += 1

        fold, gold = fcur, gcur

        if mAA == 0:
            V_new = gcur
        else:
            if mAA == 1:
                delta_f_norm = l21_norm(delta_f)
                Q[:, 0] = delta_f.flatten() / delta_f_norm
                R[0, 0] = delta_f_norm
            else:
                if mAA > mmax:
                    Q, R = qr_delete(Q, R, 1)
                    mAA -= 1

                R = np.resize(R, (mAA, mAA))
                Q = np.resize(Q, (V_len, mAA))
                for i in range(0, mAA - 1):
                    R[i, mAA - 1] = Q[:, i].T @ delta_f
                    delta_f = delta_f - (R[i, mAA - 1] * Q[:, i]).reshape((V_len, 1))

                delta_f_norm = l21_norm(delta_f)
                Q[:, mAA - 1] = delta_f.flatten() / delta_f_norm
                R[mAA - 1, mAA - 1] = delta_f_norm

            while np.linalg.cond(R) > droptol and mAA > 1:
                Q, R = qr_delete(Q, R, 1)
                mAA -= 1

            Gamma = scipy.linalg.solve(R, Q.T @ fcur.reshape(V_len, 1))
            V_new = gcur - (g @ Gamma).reshape(V.shape)

        delta_V, _ = U_converged(V_new, V_old)
        log('DELTA V', delta_V)
        V_old = V_new
        U_old = U_new
        old_E = new_E
        log('NMI', NMI(U_new, labels))
        log('E', E(U_new, V_new, X, gamma, epsilon))
        t += 1
        iterations += 1


def normal_iteration(X, C, U, V, labels, *, logger=None, multi_V=False):

    log = logger.print if logger else __builtins__.print

    t = 0
    while True:
        log('t =', t)

        delta_V = 100

        while delta_V > 1e-1:
            new_V = update_V(V, U, X, epsilon)
            delta_V = l21_norm(new_V - V)
            V = new_V
            log('DELTA V', delta_V)

            if not multi_V:
                break

        new_U = solve_U(X, V, gamma, epsilon)
        delta_U = l21_norm(U - new_U)
        U = new_U

        log('DELTA U', delta_U)
        log('E', E(U, V, X, gamma, epsilon))
        log('NMI', NMI(U, labels))

        if delta_U < 1e-1:
            log('Converged at step', t)
            log('NMI', NMI(U, labels))
            break

        t += 1

    return t, NMI(U, labels)


if __name__ == '__main__':

    from v3_tester import get_mnist_data, Logger

    logger = Logger()
    X, C, labels = get_mnist_data()
    U, V = init_uv(X, C, method='random', gamma=gamma, epsilon=epsilon)
    run(X, C, labels, logger=logger, init='preset', iter_method='aa', initial=(U, V))
    run(X, C, labels, logger=logger, init='preset', iter_method='sv', initial=(U, V))
