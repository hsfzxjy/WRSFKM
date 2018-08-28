from basics.ours import update_V, solve_U, E
from utils.math_utils import l21_norm, U_converged
from utils.metrics import nmi_acc


def iteration(X, U, V, labels, p, logger):

    epsilon, gamma = p.epsilon, p.gamma

    multi_V = p.iter_method == 'mv'

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

        metric_now = nmi_acc(U, labels)
        E_now = E(U, V, X, gamma, epsilon)
        logger.log_middle(E_now, metric_now)

        if converged:
            break

        t += 1

    return U, V, t, metric_now
