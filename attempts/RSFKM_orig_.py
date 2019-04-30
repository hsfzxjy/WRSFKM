#!/usr/bin/env python3

"""
Requires:

python-mnist
numpy
sklearn
"""

import sys
sys.path.insert(0, 'src/')


import mnist
import numpy as np
from numpy.linalg import norm as l21_norm
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import os
gamma = .1
epsilon = 30
np.random.seed(int(os.environ.get('seed', '42')))
# Download t10k_* from http://yann.lecun.com/exdb/mnist/
# Change to directory containing unzipped MNIST data
mndata = mnist.MNIST('/home/hsfzxjy/srcs/RSFKM/data/MNIST-10K/')


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


def solve_U(s, x, v, gamma):

    n = s.shape[0]

    U = np.zeros((n, C))

    for i in range(n):
        xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
        h = s[i, :] * l21_norm(xi - v, axis=1) ** 2
        h = (-h) / (2 * gamma)
        U[i, :] = solve_huang_eq_13(h)

    return U


def update_V(s, u, x):

    v = np.zeros((C, ndim))

    su = s * u

    for k in range(C):
        v[k, :] = np.average(x, axis=0, weights=su[:, k])

    return v


def update_S(x, v, epsilon, capped):

    s = np.ones((size, C))

    for i in range(len(x)):
        for k in range(C):
            norm_ = l21_norm(x[i, :] - v[k, :])
            if norm_ < epsilon:
                s[i, k] = 1 / (2 * norm_)
            else:
                s[i, k] = 0

    return s


def NMI(U):

    return nmi(labels, np.argmax(U, axis=1))


def search_U(S, X, V, gammas=(.005,)):

    return solve_U(S, X, V, gammas[0])

    # best_g, best_U, best_NMI = 0, 0, 0
    # for gamma in gammas:
    #     U = solve_U(S, X, V, gamma)
    #     # result = NMI(U)

    #     # if result > best_NMI:
    #     #     best_g = gamma
    #     #     best_U = U
    #     #     best_NMI = result

    # print(best_g, best_NMI)

    # return best_U


ndim = size = C = 0

from basics.orig._numba import solve_U, update_V, update_S


def orig(X, U, V, error, gamma=gamma):

    global ndim, size, C

    ndim = len(X[0])
    size = len(X)
    C = len(V)

    S = np.ones((size, C))
    t = 1

    while True:
        print('--- ORIG ---')
        print('== t = ', t)
        new_U = solve_U(S, X, V, gamma)
        delta = l21_norm(U - new_U)

        U = new_U
        old_V = V
        V = update_V(S, U, X)
        print('DELTA V', l21_norm(V - old_V))
        S = update_S(X, V, epsilon, True)
        # print('NMI', NMI(U))

        print('DELTA', delta)
        if delta < error and t != 1:
            print('Converged at step', t)
            break

        t += 1

    return U, V


if __name__ == '__main__':
    images, labels = mndata.load_testing()
    ndim = 784
    size = len(labels)
    C = 10
    X = np.array(images).reshape((size, ndim)) / 255
    print(np.linalg.norm(X, axis=1))

    t = 0
    V = np.random.random((C, ndim))
    U = np.zeros((size, C))

    for i in range(size):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmax(l21_norm(xi - V, axis=1))] = 1

    S = np.ones((size, C))

    while True:
        print('-------------')
        print('== t = ', t)
        new_U = solve_U(S, X, V, gamma)
        delta = l21_norm(U - new_U)
        print('DELTA U', delta)

        U = new_U
        old_V = V
        V = update_V(S, U, X)
        print('DELTA V', l21_norm(V - old_V))
        old_S = S
        S = update_S(X, V, epsilon, True)
        print('DELTA S', l21_norm(S - old_S))
        print('NMI', NMI(U))

        if delta < 1e-1:
            print('Converged at step', t)
            break

        t += 1
