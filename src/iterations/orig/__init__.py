import numpy as np

from basics.orig import update_S, update_V, solve_U, E
from utils.math_utils import U_converged
from utils.metrics import nmi_acc


def iteration(X, U, V, labels, p, logger):

    N = len(X)
    C = len(V)

    gamma, epsilon = p.gamma, p.epsilon

    S = np.ones((N, C))
    t = 0
    while True:
        new_U = solve_U(S, X, V, gamma)
        delta, converged = U_converged(new_U, U)
        print(delta)
        U = new_U
        V = update_V(S, U, X)
        S = update_S(X, V, epsilon)
        metric_now = nmi_acc(U, labels)
        E_now = E(U, V, X, gamma, epsilon)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now
