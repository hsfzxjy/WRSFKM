from utils.math_utils import U_converged
from utils.metrics import nmi_acc


from ._numba import update_U, update_V, E


def iteration(X, U, V, labels, p, logger):
    gamma = p.gamma
    t = 0
    while True:
        new_U = update_U(X, V, gamma)
        delta, converged = U_converged(new_U, U)
        print(delta)
        U = new_U
        metric_now = nmi_acc(U, labels)
        V = update_V(X, U, V, gamma)
        E_now = E(X, U, V, gamma)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now
