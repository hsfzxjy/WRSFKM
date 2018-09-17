from utils.math_utils import U_converged
from utils.metrics import nmi_acc

from ._numba import update_U, update_V, E

# def iteration(X, U, V, labels, p, logger):

#     from sklearn.cluster import KMeans

#     kmeans = KMeans(p.C, n_init=1, max_iter=200)
#     y = kmeans.fit_predict(X)

#     U = np.zeros(U.shape)
#     for i in range(len(U)):
#         U[i, y[i]] = 1

#     return U, kmeans.cluster_centers_, 99, nmi_acc(U, labels)


def iteration(X, U, V, labels, p, logger):

    t = 0
    while True:
        new_U = update_U(X, V, U)
        delta, converged = U_converged(new_U, U, tol=1e-3)
        U = new_U
        V = update_V(X, U, V)
        metric_now = nmi_acc(U, labels)
        E_now = E(X, U, V)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now
