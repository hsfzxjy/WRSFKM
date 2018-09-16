import numpy as np
from utils.math_utils import U_converged
from utils.metrics import nmi_acc
from numpy.linalg import norm as l21_norm
import math


def iteration(X, U, V, labels, p, logger):
    gamma=p.gamma
    N = len(X)
    C = len(V)
    t = 0
    while True:
        new_U = update_U(X,V,gamma)
        delta, converged = U_converged(new_U, U)
        print(delta)
        U = new_U
        metric_now = nmi_acc(U, labels)
        V=update_V(X,U,V,gamma)
        E_now = E(X,U,V,gamma)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now
    
def update_U(X,V,gamma):
    N=len(X)
    C=len(V)
    U=np.zeros((N,C))
    for i in range(N):
        for k in range(C):
            he=0
            for s in range(C):
                he=he+math.exp(-l21_norm(X[i,:]-V[s,:])/gamma)
            U[i,k]=math.exp(-l21_norm(X[i,:]-V[k,:])/gamma)/he
    return U

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
				E=E+(U[i,k])*l21_norm(X[i,:]-V[k,:])+gamma*U[i,k]*math.log(U[i,k])
	return E
