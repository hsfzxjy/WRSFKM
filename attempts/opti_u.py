#!/usr/bin/env python3

import numpy as np


def solve(p):

    n = len(p)
    I = np.identity(n)  # noqa

    mask = np.array([False] * n)
    a = 0
    while True:
        a += 1
        offset = I[mask.argmin()]
        p = p - offset

        # mat = np.concatenate((I[mask], one)).T

        mat = np.identity(n)
        mask = (~mask).reshape((n, 1))
        mat[mask @ mask.T] = 1 / mask.sum()
        # for x in mask:
        #     if not x:
        #         mat[x] = one

        # p = p - mat @ inv(mat.T @ mat) @ mat.T @ p + offset
        p = p - mat @ p + offset
        mask = p <= 0

        if mask.sum() == n - 1:
            return I[mask.argmin()]

        if (p >= 0).all():
            return p

from timeit import timeit


def solve_huang_eq_24(u):

    n = len(u)

    def f(x):
        return np.clip(x - u, 0, None).sum() / n - x

    def df(x):
        return (x > u).sum() / n - 1

    EPS = 1e-2

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


if __name__ == '__main__':
    x = np.array([1.2, 1.4, 1.3])

    data = [
        np.array([1.2, 1.4, 1.3]),
        np.array([1, 2, 1]),
        np.ones(10),
        -np.arange(100),
        np.array([1000, 0, 1000, 0])
    ]

    stmt = 'solve(x)'
    number = 10000

    for vec in data:

        print('Testing', vec)
        # print('New:', timeit(stmt, number=number, globals={'x': vec, 'solve': solve}))
        # print('Old:', timeit(stmt, number=number, globals={'x': vec, 'solve': solve_huang_eq_13}))
        print(solve_huang_eq_13(vec), solve(vec))


    # print(timeit('solve(x)', number=1000, globals={'x': x, 'solve': solve}))
    # print(timeit('solve_huang_eq_13(x)', number=1000, globals={'x': x, 'solve_huang_eq_13': solve_huang_eq_13}))

    # print(solve(x))
    # print(solve_huang_eq_13(x))
