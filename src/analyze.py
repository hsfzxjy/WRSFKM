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
        self.name = splitted[0]
        self.index = int(splitted[-1])

        f = h5py.File(filename, 'r')
        self.middle = np.array(f.get('middle'))
        self.U = np.array(f.get('U'))
        self.V = np.array(f.get('V'))
        self.result = np.array(f.get('result'))
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

        self._dataframe = pd.DataFrame([x.result for x in self.lst], columns=['step', 'nmi', 'acc'])

        return self._dataframe

    def middle(self, index):

        dct = defaultdict(list)

        for item in self.lst[index].middle:
            for i, name in enumerate(['E', 'nmi', 'acc']):
                dct[name].append(item[i])

        # self._middle = self.lst[index].middle
        return {
            k: pd.DataFrame(lst, columns=[self.name])
            for k, lst in dct.items()
        }


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

        for item_name in ['E', 'nmi', 'acc']:

            dfs = []

            for dct_ in tmp.values():
                dfs.append(dct_[item_name])

            dct[item_name] = functools.reduce(
                lambda x, y: pd.merge(x, y, how='outer', left_index=True, right_index=True),
                dfs
            )

        return dct


def load_from_directory(dir_):

    result = TestResult()

    for fn in glob(osp.join(dir_, '*.h5.*')):
        result.add_instance(Instance(fn))

    return result


def show(index):

    global result

    mid = result.middle(index)
    mid['E'].plot()
    mid['nmi'].plot()
    mid['acc'].plot()
    plt.show()


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE')

    result = load_from_directory(parser.parse_args().FILE)
    print(result.dataframe.loc['step', :])
    del parser, argparse

    import IPython
    IPython.embed()
