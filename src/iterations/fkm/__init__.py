import numpy as np
from utils.math_utils import U_converged
from utils.metrics import nmi_acc
from numpy.linalg import norm as l21_norm

from .basics import update_V, update_U, E


def iteration(X, U, V, labels, p, logger):

    # N = len(X)
    # C = len(V)
    # tol = 1e-3
    # m = p.m
    # t = 0
    # while True:
    #     V = update_V(X, U, V, m)
    #     new_U = update_U(X, V, m)
    #     delta = l21_norm(U - new_U)
    #     print(delta)
    #     U = new_U

    #     metric_now = nmi_acc(U, labels)
    #     E_now = E(X, U, V, m)
    #     if delta < tol:
    #         break
    #     logger.log_middle(E_now, metric_now)
    #     t += 1
    #
    from skfuzzy.cluster import cmeans

    V, U, _, _, _, t, _ = cmeans(X.T, len(V), 1.01, 1e-9, 100)
    metric_now = nmi_acc(U.T, labels)

    return U.T, V.T, t, metric_now


# def update_U(X, V):
#     N = len(X)
#     C = len(V)
#     U = np.zeros((N, C))
#     for i in range(N):
#         for j in range(C):
#             he = 0
#             for k in range(C):
#                 he = he + ((l21_norm(X[i, :] - V[j, :]) / l21_norm(X[i, :] - V[k, :]))**(2 / (m - 1)))
#             U[i, j] = 1 / he
#     return U


# def update_V(X, U, V):
#     N = len(X)
#     C = len(V)
#     for j in range(C):
#         fenzi = 0
#         fenmu = 0
#         for i in range(N):
#             fenzi = fenzi + (U[i, j])**m * X[i, :]
#             fenmu = fenmu + (U[i, j])**m
#         V[j, :] = fenzi / fenmu
#     return V


# def E(X, U, V):
#     N = len(X)
#     C = len(V)
#     E = 0
#     for i in range(N):
#         for k in range(C):
#             E = E + (U[i, k])**m * (l21_norm(X[i, :] - V[k, :]))**2
#     return E
