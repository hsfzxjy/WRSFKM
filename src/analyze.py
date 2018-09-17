#!/usr/bin/env python3

import h5py
from glob import glob
import os.path as osp
import numpy as np
import pandas as pd
from collections import defaultdict


class Instance:

    def __init__(self, filename):

        basename = osp.basename(filename)
        splitted = basename.split('.')
        self.index = int(splitted[-1])
        self.name = basename.rpartition('.')[0].rpartition('.')[0]

        f = h5py.File(filename, 'r')
        self.middle = np.array(f.get('middle'))
        self.U = np.array(f.get('U'))
        self.V = np.array(f.get('V'))
        self.result = np.array(f.get('result'))
        if self.result.shape == ():
            raise OSError
        f.close()


class Group:

    def __init__(self, name):

        self.lst = []
        self._dataframe = None
        self._middle = None
        self.name = name

    def add(self, instance):

        self.lst.append(instance)
        self.lst.sort(key=lambda x: x.index)

    @property
    def dataframe(self):

        if self._dataframe is not None:
            return self._dataframe

        self._dataframe = pd.DataFrame([x.result for x in self.lst], columns=['step', 'nmi', 'acc', 'time'])

        return self._dataframe

    def middle(self, index):

        dct = defaultdict(list)

        for item in self.lst[index].middle:
            for i, name in enumerate(['E', 'nmi', 'acc', 'time']):
                dct[name].append(item[i])

        # self._middle = self.lst[index].middle
        return {
            k: pd.DataFrame(lst, columns=[self.name])
            for k, lst in dct.items()
        }

    def U(self):
        return [instance.U for instance in self.lst]


class TestResult:

    def __init__(self):

        self.groups = {}
        self._dataframe = None

    def add_instance(self, instance):

        name = instance.name
        if name not in self.groups:
            self.groups[name] = Group(name)

        self.groups[name].add(instance)

    @property
    def dataframe(self):

        if self._dataframe is not None:
            return self._dataframe

        dct = {}

        for name, group in self.groups.items():
            dct[name] = group.dataframe

        self._dataframe = pd.Panel(dct).to_frame().swaplevel(0, 1)
        return self._dataframe

    def middle(self, index):

        import functools

        tmp = {}

        for name, group in self.groups.items():
            tmp[name] = group.middle(index)

        dct = {}

        for item_name in ['E', 'nmi', 'acc', 'time']:

            dfs = []

            for dct_ in tmp.values():
                dfs.append(dct_[item_name])

            dct[item_name] = functools.reduce(
                lambda x, y: pd.merge(x, y, how='outer', left_index=True, right_index=True),
                dfs
            )

        return dct

    def U(self):

        return {
            name: group.U
            for name, group in self.groups.items()
        }


def load_from_directory(dir_):

    result = TestResult()

    for fn in glob(osp.join(dir_, '*.h5.*')):
        try:
            instance = Instance(fn)
            result.add_instance(instance)
        except OSError:
            pass

    return result


def show(index):

    global result

    mid = result.middle(index)
    mid['E'].plot()
    mid['nmi'].plot()
    mid['acc'].plot()
    plt.show()


def load_from_tgz(fn):

    import tempfile
    import tarfile
    import os
    import sys

    dest = tempfile.mkdtemp()
    tar = tarfile.open(fn, 'r:gz')
    tar.extractall(dest)

    return load_from_directory(osp.join(dest, os.listdir(dest)[0]))


if __name__ == '__main__':

    import argparse
    try:
        import matplotlib.pyplot as plt
    except:  # noqa
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE')

    input = parser.parse_args().FILE

    if input.endswith('tgz') or input.endswith('tar.gz'):
        result = load_from_tgz(input)
    else:
        result = load_from_directory(input)

    print(result.dataframe.loc['step', :])
    del parser, argparse
    df = result.dataframe
    print(df.loc['acc', :].describe())
    print(df.loc['nmi', :].describe())
    print(df.loc['step', :].describe())
    print(df.loc['time', :].describe())
    import IPython
    IPython.embed()
