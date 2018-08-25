import numpy as np
from numpy.linalg import norm as l21_norm
import numba

from .math_utils_ import clip, f, df, solve_huang_eq_24, solve_huang_eq_13, solve_U, welsch_func, E, U_converged, origin_init, update_V  # noqa

# @numba.jit
# def update_V(v, u, x, epsilon):

#     N, C, ndim = len(x), len(v), len(x[0])  # noqa

#     # W = np.empty((C, N), dtype=np.float64)

#     # for i in range(N):
#     #     for k in range(C):
#     #         W[k, i] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)


#     new_v = np.zeros(v.shape)

#     for k in range(C):
#         # vec = np.empty((N,), dtype=np.float64)
#         #
#         repeated_v = np.repeat(v[k, :].reshape((1, ndim)), N, axis=0)

#         vec = u[:, k] * np.exp(
#             -epsilon *
#             l21_norm(x - repeated_v, axis=1) ** 2
#         )

#         # for i in range(N):
#         #     vec[i] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)

#         denominator = vec.sum()

#         new_v[k, :] = vec.reshape((1, N)) @ x / denominator

#     return new_v
