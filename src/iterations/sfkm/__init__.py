import numpy as np
from utils.math_utils import U_converged
from utils.metrics import nmi_acc
from numpy.linalg import norm as l21_norm
import math

from ._numba import update_V, solve_U, E


def iteration(X, U, V, labels, p, logger):

    N = len(X)
    C = len(V)

    gamma = p.gamma

    t = 0
    while True:
        new_U = solve_U(X, V, gamma)
        delta, converged = U_converged(new_U, U)
        print(delta, 'x')
        U = new_U
        V = update_V(X, U, V, gamma)
        metric_now = nmi_acc(U, labels)
        E_now = E(X, U, V, gamma)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now
