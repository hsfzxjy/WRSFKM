import numpy as np
from utils.math_utils import U_converged
from utils.metrics import nmi_acc
from numpy.linalg import norm as l21_norm
import math

def iteration(X, U, V, labels, p, logger):

    N = len(X)
    C = len(V)

    gamma= p.gamma

    t = 0
    while True:
        new_U = solve_U(X,V,gamma)
        delta, converged = U_converged(new_U, U)
        print(delta,'x')
        U = new_U
        V = update_V(X, U,V,gamma)
        metric_now = nmi_acc(U, labels)
        E_now = E(X,U,V, gamma)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now



def update_V(X,U,V,gamma):
    N=len(X)
    C=len(V)
    for j in range(C):
        fenzi=0
        fenmu=0
        for i in range(N):
            fenzi=fenzi+(U[i,j])*X[i,:]
            fenmu=fenmu+(U[i,j])
        V[j,:]=fenzi/fenmu
    return V		

def E(X,U,V,gamma):
	N=len(X)
	C=len(V)
	E=0
	for i in range(N):
		for k in range(C):
				E=E+(U[i,k])*l21_norm(X[i,:]-V[k,:])**2+gamma*U[i,k]**2
	return E


    
def solve_U(x, v, gamma):
    N=len(x)
    C=len(v)
    U = np.zeros((N, C))
    ndim=x.shape[1]
    for i in range(N):
        xi = np.repeat(x[i, :].reshape((1, ndim)), C, axis=0)#为什么要复制C次
        h = (l21_norm(xi - v, axis=1))**2
        h = (-h) / (2 * gamma)
        U[i, :] = solve_huang_eq_13(h)

    return U
    
def solve_huang_eq_13(v):
    """
    min || alpha - v ||^2, subject to \sum{alpha}=1, alpha >= 0
    """

    n = len(v)
    u = v - np.ones((n, n)) @ v / (n) + np.ones(n) / (n)
    lambda_bar_star = solve_huang_eq_24(u)
    lambda_star = (lambda_bar_star - u) + np.clip(u - lambda_bar_star, 0, None)
    return u + lambda_star - lambda_bar_star * np.ones(n)


def solve_huang_eq_24(u):

    n = len(u)

    def f(x):
        return np.clip(x - u, 0, None).sum() / n - x#这里的x会自动变成向量吗？

    def df(x):
        return (x > u).sum() / n - 1

    EPS = 1e-4

    lamb = np.min(u)
    while True:
        new_lamb = lamb - f(lamb) / df(lamb)
        if np.abs(new_lamb - lamb) < EPS:
            return new_lamb

        lamb = new_lamb
