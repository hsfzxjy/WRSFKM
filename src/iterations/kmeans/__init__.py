import numpy as np
from utils.metrics import nmi_acc


def iteration(X, U, V, labels, p, logger):

    from sklearn.cluster import KMeans

    kmeans = KMeans(p.C, n_init=1, max_iter=200)
    y = kmeans.fit_predict(X)

    U = np.zeros(U.shape)
    for i in range(len(U)):
        U[i, y[i]] = 1

    return U, kmeans.cluster_centers_, 99, nmi_acc(U, labels)
