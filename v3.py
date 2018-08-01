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

gamma = .001
epsilon = 1

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


from opti_u import solve as solve_huang_eq_13_new
import pymp
import multiprocessing as mp


# old_v deprecated
def solve_U(x, v, old_v, gamma):

    N = len(x)
    C = len(v)
    ndim = len(x[0])
    U = pymp.shared.array((N, C))

    with pymp.Parallel(mp.cpu_count()) as p:
        for i in p.range(N):
            xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
            # norm_v = l21_norm(xi - v, axis=1)
            # norm_v_old = l21_norm(xi - old_v, axis=1)
            # W = welsch_func(norm_v_old)
            # h = W + (norm_v - norm_v_old) * (1 - epsilon * W)
            h = welsch_func(l21_norm(xi - v, axis=1))
            h = (-h) / (2 * gamma)
            U[i, :] = solve_huang_eq_13(h)
            # U[i, :] = solve_huang_eq_13_new(h)

    return U


def update_V(v, u, x):

    N = len(x)
    C = len(v)
    ndim = len(x[0])
    W = np.zeros((N, C))

    for i in range(N):
        for k in range(C):
            W[i, k] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)

    new_v = np.zeros(v.shape)

    for k in range(C):
        denominator = W[:, k].sum()

        # # Avoid division by zero
        # if denominator == 0:
        #     denominator = 1

        new_v[k, :] = W[:, k].reshape((1, N)) @ x / denominator

    return new_v


def NMI(U):

    return nmi(labels, np.argmax(U, axis=1))


def init_uv(X):

    N = len(X)

    V = _init_centroids(X, C, 'k-means++')

    # V = np.random.random((C, ndim))
    U = np.ones((size, C)) * .1 / (C - 1)

    for i in range(size):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = .9

    return U, V


def run_new(U, V, log_file_name):

    log_file = open(log_file_name + '.new', 'w')

    t = 0

    print = lambda *args, **kwargs: print(*args, **kwargs, file=log_file)  # noqa

    while True:
        print('-------------')
        print('== t = ', t)

        delta_V = 100

        while delta_V > 1e-1:
            new_V = update_V(V, U, X)
            delta_V = l21_norm(new_V - V)
            V = new_V
            print('DELTA V', delta_V)

        new_U = solve_U(X, V, V, gamma)
        delta_U = l21_norm(U - new_U)
        U = new_U

        print('DELTA U', delta_U)
        print('NMI', NMI(U))

        if delta_U < 1e-1:
            print('Converged at step', t)
            print('NMI', NMI(U))
            break

        t += 1

    log_file.close()
    return t, NMI(U)


def run_old(U, V, log_file_name):

    log_file = open(log_file_name + '.old', 'w')

    t = 0

    print = lambda *args, **kwargs: print(*args, **kwargs, file=log_file)  # noqa

    while True:
        print('-------------')
        print('== t = ', t)

        delta_V = 100

        new_V = update_V(V, U, X)
        delta_V = l21_norm(new_V - V)
        V = new_V
        print('DELTA V', delta_V)

        new_U = solve_U(X, V, V, gamma)
        delta_U = l21_norm(U - new_U)
        U = new_U

        print('DELTA U', delta_U)
        print('NMI', NMI(U))

        if delta_U < 1e-1:
            print('Converged at step', t)
            print('NMI', NMI(U))
            break

        t += 1

    log_file.close()
    return t, NMI(U)


if __name__ == '__main__':
    from sklearn.cluster.k_means_ import _init_centroids, _tolerance
    from RSFKM_orig import orig

    images, labels = mndata.load_testing()
    ndim = 784
    N = size = len(labels)
    C = 10
    X = np.array(images).reshape((size, ndim)) / 255
    t = 0

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    action = parser.parse_args().action

    if action == 'test':
        for i in range(200):
            U, V = init_uv(X)
            fn = 'v3_log/{}.log'.format(i)
            t1, nmi1 = run_new(U, V, fn)
            t2, nmi2 = run_old(U, V, fn)

            with open('v3_log/{}.stat'.format(i), 'w') as f:
                print(t1, nmi1, file=f)
                print(t2, nmi2, file=f)
