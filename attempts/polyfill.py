def solve_huang_eq_24(u):

    n = len(u)

    def f(x):
        return np.clip(x - u, 0, None).sum() / n - x

    def df(x):
        return (x > u).sum() / n - 1

    EPS = 1e-4

    lamb = np.min(u)
    while True:
        new_lamb = lamb - f(lamb) / df(lamb)
        if np.abs(new_lamb - lamb) < EPS:
            return new_lamb

        lamb = new_lamb


def solve_huang_eq_13(v):
    """
    min || alpha - v ||^2, subject to \sum{alpha}=1, alpha >= 0
    """

    n = len(v)
    u = v - np.ones((n, n)) @ v / (n) + np.ones(n) / (n)
    lambda_bar_star = solve_huang_eq_24(u)
    lambda_star = (lambda_bar_star - u) + np.clip(u - lambda_bar_star, 0, None)
    return u + lambda_star - lambda_bar_star * np.ones(n)


def welsch_func(x):

    result = (1 - np.exp(- epsilon * x ** 2)) / epsilon

    return result


# from opti_u import solve as solve_huang_eq_13_new
# import pymp
# import multiprocessing as mp


def solve_U(x, v, gamma):

    N, C, ndim = len(x), len(v), len(x[0])

    U = pymp.shared.array((N, C))

    with pymp.Parallel(mp.cpu_count()) as p:
        for i in p.range(N):
            xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
            h = welsch_func(l21_norm(xi - v, axis=1))
            h = (-h) / (2 * gamma)
            U[i, :] = solve_huang_eq_13(h)

    return U


def update_V(v, u, x):

    N, C, ndim = len(x), len(v), len(x[0])  # noqa

    W = np.zeros((N, C))

    for i in range(N):
        for k in range(C):
            W[i, k] = u[i, k] * np.exp(-epsilon * l21_norm(x[i, :] - v[k, :])**2)

    new_v = np.zeros(v.shape)

    for k in range(C):
        denominator = W[:, k].sum()

        new_v[k, :] = W[:, k].reshape((1, N)) @ x / denominator

    return new_v


def origin_init(X, C):

    N, ndim = len(X), len(X[0])
    size = N

    def origin_solve_U(x, v, gamma):
        U = np.zeros((N, C))
        for i in range(N):
            xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)
            h = l21_norm(xi - v, axis=1)
            h = (-h) / (4 * gamma * epsilon**0.5 / 0.63817562)
            U[i, :] = solve_huang_eq_13(h)
        return U

    def origin_update_V(x, u, v):
        V = np.zeros((C, ndim))
        for k in range(C):
            A = 0
            vk = v[k, :].reshape((1, ndim))
            for i in range(N):
                xi = x[i, :].reshape((1, ndim))
                V[k, :] = V[k, :] + 1 / (2 * l21_norm(xi - vk)) * u[i, k] * xi
                A = A + 1 / (2 * l21_norm(xi - vk)) * u[i, k]
            V[k, :] = V[k, :] / A
        return V

    V = np.random.random((C, ndim))
    U = np.zeros((size, C))
    while True:
        U = origin_solve_U(X, V, gamma)
        new_V = origin_update_V(X, U, V)
        delta = l21_norm(new_V - V)
        V = new_V
        print(delta)
        if delta < 1e-1:
            break

    return U, V
