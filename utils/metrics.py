import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from collections import defaultdict, Counter


def acc(X, Y):

    mapping = defaultdict(list)

    N = len(X)

    for x, y in zip(X, Y):
        mapping[y].append(x)

    corrects = 0

    for lst in mapping.values():
        corrects += Counter(lst).most_common(1)[0][-1]

    return corrects / N


def metric(U, labels):

    X = np.argmax(U, axis=1)

    return nmi(X, labels), acc(X, labels)
