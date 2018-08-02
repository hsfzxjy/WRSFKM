#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob


def load_stats():

    ts, nmis = [], []

    for fn in glob('v3_log/*.stat'):
        with open(fn, 'r') as f:
            new_t, new_nmi = map(float, f.readline().strip().split())
            old_t, old_nmi = map(float, f.readline().strip().split())
            ts.append([new_t, old_t])
            nmis.append([new_nmi, old_nmi])

    ts = pd.DataFrame(data=ts, columns=['new steps', 'old steps'])
    nmis = pd.DataFrame(data=nmis, columns=['new NMI', 'old NMI'])

    return ts, nmis


ts, nmis = load_stats()
print(ts.describe())
print(nmis.describe())
ts.hist()
nmis.hist()
# plt.plot(nmis[:, 0] - nmis[:, 1])
# plt.grid()
plt.show()
# plt.plot(ts[:, 0] - ts[:, 1])
# plt.grid()
# plt.show()
