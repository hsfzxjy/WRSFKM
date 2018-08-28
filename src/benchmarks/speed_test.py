#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',))

from timeit import timeit


def data():

    from init import init_uv
    from utils.datasets import load_dataset
    from utils.params import Params

    X, C, labels = load_dataset('mnist_10k')

    return (
        X,
        *init_uv(
            X, C,
            Params({
                'method': 'random'
            })
        ),
        labels
    )


from basics.ours._numba import solve_U as numba_solve_U, update_V as numba_update_V, E as numba_E
from basics.ours._cython import update_V as cython_update_V
from utils.metrics import nmi_acc

functions = {
    'numba_solve_U': lambda X, U, V, _: numba_solve_U(X, V, 0.7, 1e-1),
    'numba_update_V': lambda X, U, V, _: numba_update_V(V, U, X, 1e-1),
    'cython_update_V': lambda X, U, V, _: cython_update_V(V, U, X, 1e-1),
    'numba_E': lambda X, U, V, _: numba_E(U, V, X, 0.7, 1e-1),
    'nmi_acc': lambda X, U, V, labels: nmi_acc(U, labels),
}


if __name__ == '__main__':

    print('Testing on MNIST-10K...')

    X, U, V, labels = data()

    for name, function in functions.items():
        print('!!!!', name, timeit(
            stmt='func(X, U, V, labels)',
            globals={
                'X': X,
                'U': U,
                'V': V,
                'labels': labels,
                'func': function
            },
            number=5
        ), sep='\t')
