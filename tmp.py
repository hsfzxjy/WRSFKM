#!/usr/bin/env python3

import numpy as np
from numpy import concatenate as cat


def make(n, indices, fuck_one=True):

    lst = []
    one = np.ones(n)
    for x in indices:
        a = [0] * n
        a[x] = 1
        lst.append(a)
        if fuck_one:
            one[x] = 0

    # one[indices] = 0
    lst.append(one)
    return np.array(lst)

indices = (5, 3, 2)

# A = make(6, indices).T
B = make(6, indices, False)
# print(A, B, sep='\n')
# print(A @ B)
# print('---')
from numpy.linalg import inv
print(B.T @ inv(B @ B.T) @ B)

mask = np.array([False] * 6).reshape((1, 6))
for x in indices:
    mask[(0, x)] = True

I = np.identity(6)

I[(~mask).T @ (~mask)] = 4
print(I)
