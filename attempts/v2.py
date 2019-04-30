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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--orig-gamma', type=float)
parser.add_argument('--epsilon', type=float, default=0.03)
parser.add_argument('--orig-epsilon', type=float)
parser.add_argument('--seed', type=int)
args = parser.parse_args()


# np.random.seed(int(os.environ.get('seed', '42')))
# print('Using seed:', os.environ.get('seed', '42'))

epsilon = args.epsilon
gamma = args.orig_gamma / args.orig_epsilon / epsilon
np.seed(args.seed)

# epsilon = 0.03
# gamma = .2 / 30 / epsilon
# np.random.seed(42)

# Download t10k_* from http://yann.lecun.com/exdb/mnist/
# Change to directory containing unzipped MNIST data
mndata = mnist.MNIST('data/MNIST-10K/')


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


import pymp
import multiprocessing as mp


def solve_U(x, v, old_v, gamma):

    U = pymp.shared.array((N, C))

    with pymp.Parallel(min(mp.cpu_count(), 4)) as p:
        for i in p.range(N):
            xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
            norm_v = l21_norm(xi - v, axis=1)
            norm_v_old = l21_norm(xi - old_v, axis=1)
            W = welsch_func(norm_v_old)
            h = W + (norm_v - norm_v_old) * (1 - epsilon * W)
            # h = welsch_func(l21_norm(xi - v, axis=1))
            h = (-h) / (2 * gamma)
            U[i, :] = solve_huang_eq_13(h)
            # U[i, :] = solve_huang_eq_13_new(h)

    return U


def update_V(v, u, x):

    W = np.zeros((N, C))

    for i in range(N):
        for k in range(C):
            W[i, k] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)

    new_v = np.zeros(v.shape)

    for k in range(C):
        denominator = W[:, k].sum()
        new_v[k, :] = W[:, k].reshape((1, N)) @ x / denominator

    return new_v


from basics.ours._numba import E

def target(U, V, X):

    return E(U, V, X, gamma, epsilon)

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
    U = np.ones((size, C)) * .1 / (C - 1)

    for i in range(size):
        xi = np.repeat(X[i, :].reshape((1, ndim)), C, axis=0)
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = 1

    S = np.ones((size, C))

    delta_U = 10
    while delta_U > 0.1:
        print('-------------')
        print('== t = ', t)

        delta_U = 100

        old_V = V.copy()

        # while delta_U > 1:
        for _ in range(1):
            new_V = update_V(old_V, U, X)
            delta_V = l21_norm(new_V - V)
            V = new_V

            new_U = solve_U(X, V, old_V, gamma)
            delta_U = l21_norm(U - new_U)
            U = new_U

            print('DELTA V', delta_V)
            print('DELTA U', delta_U)
            print('NMI', NMI(U))
            print(target(U, V, X))

        # if delta_V < .1:
        #     print('Converged at step', t)
        #     print('NMI', NMI(U))
        #     break

        t += 1
