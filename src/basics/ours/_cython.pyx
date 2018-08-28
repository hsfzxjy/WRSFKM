from libc.math cimport exp

import numpy as np
cimport numpy as np
from numpy.linalg import norm

cimport cython
from cython.parallel import prange

py_float = np.float64
ctypedef np.float64_t FLOAT
ctypedef np.float64_t[:] FLOAT1D
ctypedef np.float64_t[:, :] FLOAT2D

cdef FLOAT welsch_func(FLOAT x, FLOAT epsilon):

    return (1 - np.exp(- epsilon * x * x)) / epsilon

@cython.boundscheck(False)
cdef inline FLOAT sqrdistance(FLOAT1D x, FLOAT1D y) nogil:

    cdef FLOAT result = 0.
    cdef Py_ssize_t i
    cdef FLOAT delta

    for i in range(x.shape[0]):
        delta = x[i] - y[i]
        result += delta * delta

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_V(FLOAT2D V, FLOAT[:, ::1] U, FLOAT2D X, FLOAT epsilon):

    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t ndim = X.shape[1]
    cdef Py_ssize_t C = V.shape[0]

    new_v = np.zeros((C, ndim), dtype=py_float)
    cdef FLOAT[:, ::1] new_v_view = new_v.view()

    cdef FLOAT[::1, :] Ut = U.T

    cdef FLOAT denominator
    cdef FLOAT tmp
    cdef Py_ssize_t i, j, k


    vec = np.zeros((N, ), dtype=py_float)
    cdef FLOAT[::1] vec_view = vec.view()

    for k in range(C):
        denominator = 0

        for i in prange(N, nogil=True):
            tmp = vec_view[i] = U[i, k] * exp(
                -epsilon *
                sqrdistance(X[i, :], V[k, :])
            )
            denominator += tmp

        for j in prange(ndim, nogil=True):

            tmp = 0.
            for i in range(N):
                tmp = tmp + vec_view[i] * X[i, j]
            new_v_view[k, j] = tmp / denominator

    return new_v
