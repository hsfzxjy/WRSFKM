from utils.math_utils import U_converged
from utils.metrics import nmi_acc

from ._numba import update_V, update_U, E


# def iteration(X, U, V, labels, p, logger):

#     from skfuzzy.cluster import cmeans

#     V, U, _, _, _, t, _ = cmeans(X.T, len(V), 1.01, 1e-9, 100)
#     metric_now = nmi_acc(U.T, labels)

#     return U.T, V.T, t, metric_now


def iteration(X, U, V, labels, p, logger):

    m = p.m
    t = 0
    while True:
        new_U = update_U(X, V, m)
        delta, converged = U_converged(U, new_U)
        U = new_U

        V = update_V(X, U, V, m)
        metric_now = nmi_acc(U, labels)
        E_now = E(X, U, V, m)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now
