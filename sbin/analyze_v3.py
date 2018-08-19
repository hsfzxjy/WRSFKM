#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import tarfile
from collections import defaultdict


def load_stats(fn, num=300):

    tdf, nmidf = defaultdict(list), defaultdict(list)

    tar = tarfile.open(fn, 'r:gz')

    for member in tar.getmembers():

        if not member.name.endswith('stat'):
            continue

        f = tar.extractfile(member)
        name = member.name.rpartition('.')[0]
        for line in f:
            t, nmi = map(float, line.strip().split())
            tdf[name].append(t)
            nmidf[name].append(nmi)

        missing = num - len(tdf[name])

        if missing > 0:
            tdf[name] += [np.nan] * missing
            nmidf[name] += [np.nan] * missing

    return pd.DataFrame(data=tdf), pd.DataFrame(data=nmidf)


def load_stats_2(fn, num=300):

    import re

    tdf, nmidf = defaultdict(dict), defaultdict(dict)

    tar = tarfile.open(fn, 'r:gz')

    for member in tar.getmembers():

        if 'log' not in member.name:
            continue

        f = tar.extractfile(member)
        name = member.name.partition('.')[0]
        index = int(member.name.rpartition('.')[-1])

        t = nmi = 0
        for line in f:

            result = re.findall(rb'NMI ([\d\.]+)', line)

            if result:
                nmi = float(result[0])

            result = re.findall(rb't = (\d+)', line)

            if result:
                t = int(result[0])

        tdf[name][index] = t
        nmidf[name][index] = nmi

    return pd.DataFrame(data=tdf), pd.DataFrame(data=nmidf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE')

    fn = parser.parse_args().FILE

    t, nmi = load_stats_2(fn)

    print(t.describe())
    print(nmi.describe())
    delta = (t['v4_result/sv_random'] - t['v4_result/aa_random'])
    print(delta.describe())
    delta.plot(style='-')
    delta_nmi = (nmi['v4_result/sv_random'] - nmi['v4_result/aa_random'])
    print(delta_nmi.describe())
    delta_nmi.plot(style='-')
    t.hist()
    nmi.hist()

    plt.show()

    for col in nmi:
        print(col)
        print(nmi[col].nlargest(30))

    for col in t:
        plt.plot(t[col], nmi[col], 'o')

        plt.show()

    # # plt.plot(nmis[:, 0] - nmis[:, 1])
    # # plt.grid()
    # df.plot(x='new steps', y='new NMI', style='.')
    # df.plot(x='old steps', y='old NMI', style='.')

    # # plt.plot(ts[:, 0] - ts[:, 1])
    # # plt.grid()
    # # plt.show()
