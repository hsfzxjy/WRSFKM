#!/usr/bin/env python3

"""
Requires:

python-mnist
numpy
sklearn

Anderson Acceleration
"""

import os
import numpy as np
import scipy
from numpy.linalg import norm as l21_norm
from sklearn.cluster.k_means_ import _init_centroids
from scipy.linalg import qr_delete
from utils.math_utils import E
from utils.metrics import metric
from time import time

from utils.math_utils import solve_U, update_V, origin_init, U_converged

# gamma = .001
# epsilon = 1e-4


def init_uv(X, C, p):

    N, ndim = len(X), len(X[0])

    np.random.seed()

    print(p, 'test seed', np.random.random((1,)))
    assert isinstance(p.method, str)

    if p.method == 'random':
        V = np.random.random((C, ndim))
    elif p.method == 'orig':
        return origin_init(X, C, p.gamma, p.epsilon)
    else:
        V = _init_centroids(X, C, p.method)

    U = np.ones((N, C)) * .1 / (C - 1)

    for i in range(N):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = .9

    return U, V


def run(X, labels, p, logger):

    C = p.C

    assert isinstance(p.iter_method, str)

    if p.init == 'preset':
        U, V = p.initial
    else:
        U, V = init_uv(X, C, method=p.init)

    iter_method = p.iter_method.lower()

    if iter_method in ['sv', 'mv']:
        return normal_iteration(X, U, V, labels, p, logger=logger)
    elif iter_method == 'aa':
        return anderson_iteration(X, U, V, labels, p, logger=logger)
    elif iter_method == 'orig':
        from orig import orig_iteration

        return orig_iteration(X, U, V, labels, p, logger=logger)


from anderson import anderson_iteration  # noqa


def normal_iteration(X, U, V, labels, p, logger):

    epsilon, gamma = p.epsilon, p.gamma

    multi_V = p.iter_method == 'mv'

    # energy = Energy()

    t = 0
    while True:

        delta_V = 100

        while delta_V > 1e-1:
            new_V = update_V(V, U, X, epsilon)
            delta_V = l21_norm(new_V - V)
            V = new_V

            if not multi_V:
                break

        new_U = solve_U(X, V, gamma, epsilon)
        _, converged = U_converged(new_U, U)
        U = new_U

        metric_now = metric(U, labels)
        E_now = E(U, V, X, gamma, epsilon)
        logger.log_middle(E_now, metric_now)

        # energy.add(E_now)

        if converged:
            break

        t += 1

    return U, V, t, metric_now


if __name__ == '__main__':

    from tester import DualTester

    # DualTester(
    #     root_directory='',
    #     init_params=dict(method='random'),
    #     mutual={'epsilon': 0.005, 'gamma': .05},
    #     dataset='mnist_10k',
    #     params={
    #         'aa_random': {'iter_method': 'aa', 'mmax': 4},
    #         # 'sv_random': {'iter_method': 'sv'},
    #     },
    #     times=1
    # ).execute()
    #
    DualTester(
        root_directory='',
        init_params=dict(method='random'),
        mutual={'epsilon': 0.005, 'gamma': .01},
        dataset='coil_20',
        params={
            'aa_random': {'iter_method': 'aa', 'mmax': 4},
            # 'sv_random': {'iter_method': 'sv'},
        },
        times=1
    ).execute()

    # # # from v3_tester import get_mnist_data, Logger

    # # # Download t10k_* from http://yann.lecun.com/exdb/mnist/
    # # # Change to directory containing unzipped MNIST data
    # # mndata = mnist.MNIST('./MNIST/')

    # logger = Logger()
    # X, C, labels = get_mnist_data()
    # U, V = init_uv(X, C, method='random', gamma=gamma, epsilon=epsilon)
    # run(X, C, labels, logger=logger, init='preset', iter_method='aa', initial=(U, V))
    # run(X, C, labels, logger=logger, init='preset', iter_method='sv', initial=(U, V))
