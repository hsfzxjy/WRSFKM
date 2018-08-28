import os

M = os.environ.get('M', 'numba')

if M == 'numba':
    from ._numba import update_V, solve_U, E  # noqa
