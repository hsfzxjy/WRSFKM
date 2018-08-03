#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import tarfile


def load_stats(fn):

    # ts, nmis = [], []
    df = []

    tar = tarfile.open(fn, 'r:gz')

    for member in tar.getmembers():

        if not member.name.endswith('stat'):
            continue

        f = tar.extractfile(member)

        new_t, new_nmi = map(float, f.readline().strip().split())
        old_t, old_nmi = map(float, f.readline().strip().split())
        df.append([new_t, old_t, new_nmi, old_nmi])

    return pd.DataFrame(data=df, columns=['new steps', 'old steps', 'new NMI', 'old NMI'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE')

    fn = parser.parse_args().FILE

    df = load_stats(fn)
    print(df.describe())
    df.hist()
    # plt.plot(nmis[:, 0] - nmis[:, 1])
    # plt.grid()
    df.plot(x='new steps', y='new NMI', style='.')
    df.plot(x='old steps', y='old NMI', style='.')

    plt.show()
    # plt.plot(ts[:, 0] - ts[:, 1])
    # plt.grid()
    # plt.show()
