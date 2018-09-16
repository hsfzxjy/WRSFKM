from utils.math_utils import U_converged
from utils.metrics import nmi_acc
from basics.AFKM import update_V,solve_U,E

def iteration(X,U,V,labels,p,logger):

	gamma = p.gamma
	t = 0
	while True:
		new_U = solve_U(X, V, gamma)
		converged = U_converged(new_U, U)
		U = new_U
		V =update_V(U, X)
		metric_now = nmi_acc(U, labels)
        E_now = E(U, V, X, gamma)
        logger.log_middle(E_now, metric_now)

        if converged:
        	break

        t += 1

    return U, V, t, metric_now
