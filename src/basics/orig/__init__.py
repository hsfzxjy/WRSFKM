from ._numba import update_S, solve_U, E, update_V  # noqa


# def update_V(s, u, x):

#     _, C = s.shape
#     N, ndim = x.shape

#     v = np.zeros((C, ndim))

#     su = s * u

#     for k in range(C):
#         v[k, :] = np.average(x, axis=0, weights=su[:, k])

#     return v
