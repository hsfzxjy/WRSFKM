#!/usr/bin/env python3

"""
Requires:

python-mnist
numpy
sklearn

Anderson Acceleration
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np
import scipy
from numpy.linalg import norm as l21_norm
from sklearn.cluster.k_means_ import _init_centroids
from scipy.linalg import qr_delete
from math_utils import E
from metrics import metric

from math_utils import solve_U, update_V, origin_init, U_converged

# gamma = .001
# epsilon = 1e-4


def init_uv(X, C, p):

    N, ndim = len(X), len(X[0])

    np.random.seed(os.getpid())

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

    if p.iter_method.lower() in ['sv', 'mv']:
        return normal_iteration(X, U, V, labels, p, logger=logger)
    elif p.iter_method.lower() == 'aa':
        return anderson_iteration(X, U, V, labels, p, logger=logger)


def anderson_iteration(X, U, V, labels, p, logger):

    V_len = V.flatten().shape[0]

    mAA = 0
    V_old = V
    U_old = U

    iterations = t = 0
    mmax = p.mmax or 4
    AAstart = p.AAstart or 0
    droptol = p.droptol or 1e10

    gamma = p.gamma
    epsilon = p.epsilon
    max_iteration = p.max_iterations or 300

    fold = gold = None
    g = np.ndarray(shape=(V_len, 0))
    Q = np.ndarray(shape=(V_len, 1))
    R = np.ndarray(shape=(1, 1))

    old_E = np.Infinity
    VAUt = None

    while True:
        U_new = solve_U(X, V_old, gamma, epsilon)

        delta_U, is_converged = U_converged(U_new, U_old)

        new_E = E(U_new, V_old, X, gamma, epsilon)

        if is_converged:
            return U_new, V_old, t, metric(U_new, labels)

        if t > max_iteration:
            return U_new, V_old, t, metric(U_new, labels)

        if new_E >= old_E:
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
        V_old = V_new
        U_old = U_new
        old_E = new_E
        logger.log_middle(E(U_new, V_new, X, gamma, epsilon), metric(U_new, labels))
        t += 1
        iterations += 1


from anderson import anderson_iteration  # noqa
from energy import Energy


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

    DualTester(
        root_directory='',
        init_params=dict(method='random'),
        mutual={'epsilon': 1e-4, 'gamma': .1},
        dataset='mnist_10k',
        params={
            'aa_random': {'iter_method': 'aa', 'mmax': 3},
            'sv_random': {'iter_method': 'sv'},
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
