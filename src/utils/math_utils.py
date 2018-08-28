from numpy.linalg import norm as l21_norm


def U_converged(old, new):

    tol = 1e-1

    delta = l21_norm(old - new)

    return delta, delta < tol
