#!/usr/bin/env python3

"""
Requires:

python-mnist
numpy
sklearn
"""

import mnist
import numpy as np
from numpy.linalg import norm as l21_norm
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

gamma = .005
epsilon = 1e-5

# Download t10k_* from http://yann.lecun.com/exdb/mnist/
# Change to directory containing unzipped MNIST data
mndata = mnist.MNIST('/data/science/MNIST/')


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


def solve_U(x, v, gamma):

    U = np.zeros((N, C))

    for i in range(N):
        xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
        h = welsch_func(l21_norm(xi - v, axis=1))
        h = (-h) / (2 * gamma)
        U[i, :] = solve_huang_eq_13(h)

    return U


def update_V(v, u, x):

    W = np.zeros((N, C))

    for i in range(N):
        for k in range(C):
            W[i, k] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)

    new_v = np.zeros(v.shape)

    for k in range(C):
        denominator = W[:, k].sum()

        # Avoid division by zero
        if denominator == 0:
            denominator = 1

        new_v[k, :] = W[:, k].reshape((1, N)) @ x / denominator

    return new_v


def NMI(U):

    return nmi(labels, np.argmax(U, axis=1))


if __name__ == '__main__':
    images, labels = mndata.load_testing()
    ndim = 784
    N = size = len(labels)
    C = 10
    X = np.array(images).reshape((size, ndim)) / 255

    t = 0
    V = np.random.random((C, ndim))
    U = np.zeros((size, C))

    for i in range(size):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = 1

    S = np.ones((size, C))


    while True:
        print('-------------')
        print('== t = ', t)
        new_V = update_V(V, U, X)
        delta_V = l21_norm(new_V - V)
        V = new_V
        print('DELTA', delta_V)

        delta = 100

        while delta > 1:
            new_U = solve_U(X, V, gamma)
            delta = l21_norm(U - new_U)

            U = new_U

        print('NMI', NMI(U))

        if delta_V < 1e-1:
            print('Converged at step', t)
            print('NMI', NMI(U))
            break

        t += 1
