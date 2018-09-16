import numpy as np
from utils.math_utils import U_converged
from utils.metrics import nmi_acc
from numpy.linalg import norm as l21_norm


def iteration(X, U, V, labels, p, logger):

    N = len(X)
    C = len(V)

    t = 0
    while True:
        new_U = update_U(X,V,U)
        delta, converged = U_converged(new_U, U)
        print(delta)
        U = new_U
        V = update_V(X,U,V)
        metric_now = nmi_acc(U, labels)
        E_now = E(X,U,V)
        if converged:
            break
        logger.log_middle(E_now, metric_now)
        t += 1

    return U, V, t, metric_now
    
def update_U(X,V,U):
	N=len(X)
	C=len(V)
	U=np.zeros((N,C))
	for i in range(N):
		tmp=0
		for k in range(C):
			if l21_norm(X[i,:]-V[k,:])<l21_norm(X[i,:]-V[tmp,:]):
				tmp=k
		U[i,tmp]=1
	return U

def update_V(X,U,V):
	N=len(X)
	C=len(V)
	for i in range(C):
		count=0
		sum=0
		for k in range(N):
			if U[k,i]==1:
				count=count+1
				sum=sum+X[k,:]
		if count!=0:
			V[i,:]=sum/count
		
	return V		

def E(X,U,V):
	N=len(X)
	C=len(V)
	E=0
	for i in range(N):
		for k in range(C):
			if U[i,k]==1:
				E=E+(l21_norm(X[i,:]-V[k,:]))**2
	return E
