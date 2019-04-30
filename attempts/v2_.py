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
np.random.seed(int(os.environ.get('seed', '42')))
print('Using seed:', os.environ.get('seed', '42'))

epsilon = 0.03
gamma = .1 / 30 / epsilon
# np.random.seed(42)

# Download t10k_* from http://yann.lecun.com/exdb/mnist/
# Change to directory containing unzipped MNIST data
mndata = mnist.MNIST('data/MNIST-10K/')


def welsch_func(x):

    result = (1 - np.exp(- epsilon * x ** 2)) / epsilon

    return result


from basics.ours._numba import E, solve_U, update_V

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
        U[i, np.argmin(l21_norm(xi - V, axis=1))] = .9

    S = np.ones((size, C))

    delta_U = 10
    while delta_U > 0.1:
        print('-------------')
        print('== t = ', t)

        delta_U = 100

        old_V = V.copy()

        new_V = update_V(old_V, U, X, epsilon)
        delta_V = l21_norm(new_V - V)
        V = new_V

        new_U = solve_U(X, V, old_V, gamma, epsilon)
        delta_U = l21_norm(U - new_U)
        U = new_U

        print('DELTA V', delta_V)
        print('DELTA U', delta_U)
        print('NMI', NMI(U))
        print(target(U, V, X))

        t += 1
