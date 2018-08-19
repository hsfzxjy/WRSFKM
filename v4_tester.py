#!/usr/bin/env python3

import sys
import os


def get_mnist_data():

    import mnist
    import numpy as np

    images, labels = mnist.MNIST('./MNIST/').load_testing()
    ndim = 784
    size = len(labels)
    C = 10
    X = np.array(images).reshape((size, ndim)) / 255

    return X, C, labels


class Logger:

    def __init__(self, filename=None):

        if filename is None:
            self._fd = sys.stdout
        else:
            self._fd = open(filename, 'w')

    def print(self, *args, **kwargs):

        print(*args, file=self._fd, **kwargs)
        self._fd.flush()

    def close(self):

        if self._fd is not sys.stdout:
            self._fd.close()


class Tester:

    def __init__(self, *, name, root_directory, params, times):

        self.name = name
        self.root_directory = root_directory
        self.params = params
        self.times = times

        os.makedirs(root_directory, exist_ok=True)

    def execute(self):

        from v3 import run

        X, C, labels = get_mnist_data()
        main_logger = Logger(os.path.join(self.root_directory, self.name + '.stat'))

        for i in range(self.times):

            logger = Logger(os.path.join(self.root_directory, self.name + '.log.' + str(i)))
            t, nmi = run(X, C, labels, logger=logger, **self.params)
            logger.close()

            main_logger.print(t, nmi)

        main_logger.close()


import multiprocessing
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class GridTester:

    def __init__(self, root_directory, times, param_groups):

        self.param_groups = param_groups
        self.root_directory = root_directory
        self.times = times

    def execute(self):

        Pool(
            processes=min(
                len(self.param_groups),
                os.cpu_count() // 2
            ),
        ).map(self._target, self.param_groups)

    def _target(self, param_group):
        name, params = param_group

        Tester(
            name=name,
            root_directory=self.root_directory,
            times=self.times,
            params=params,
        ).execute()


class DualTester:

    def __init__(self, *, root_directory, init_params, params, times):

        self.root_directory = root_directory
        self.params = params
        self.init_params = init_params
        self.times = times

        os.makedirs(root_directory, exist_ok=True)

    def execute(self):

        from v4 import run, init_uv

        X, C, labels = get_mnist_data()

        main_loggers = {
            name: Logger(os.path.join(self.root_directory, name + '.stat'))
            for name in self.params
        }

        for i in range(self.times):
            print('times', i)
            U, V = init_uv(X, C, **self.init_params)

            for name, param in self.params.items():

                logger = Logger(os.path.join(self.root_directory, name + '.log.' + str(i)))
                t, nmi = run(X, C, labels, logger=logger, init='preset', initial=(U, V), **param)
                logger.close()

                main_loggers[name].print(t, nmi)

        for logger in main_loggers.values():
            logger.close()


if __name__ == '__main__':

    DualTester(
        root_directory='v4_result',
        init_params=dict(method='random'),
        params={
            'sv_random': {'iter_method': 'sv'},
            'aa_random': {'iter_method': 'aa'},
        },
        times=200
    ).execute()

    # DualTester(
    #     root_directory='v4_result',
    #     init_params=dict(method='orig'),
    #     params={
    #         'sv_orig': {'iter_method': 'sv'},
    #         'aa_orig': {'iter_method': 'aa'},
    #     },
    #     times=200,
    # ).execute()

    # pg1 = [
    #     (
    #         'sv_random',
    #         {
    #             'multi_V': False,
    #             'init': 'random'
    #         }
    #     ),
    #     (
    #         'sv_kmpp',
    #         {
    #             'multi_V': False,
    #             'init': 'k-means++'
    #         }

    #     ),
    #     (
    #         'mv_random',
    #         {
    #             'multi_V': True,
    #             'init': 'random'
    #         }
    #     ),
    #     (
    #         'mv_kmpp',
    #         {
    #             'multi_V': True,
    #             'init': 'k-means++'
    #         }
    #     ),
    # ]

    # pg2 = [
    #     (
    #         'mv_orig',
    #         {
    #             'multi_V': True,
    #             'init': 'orig'
    #         }
    #     ),
    #     (
    #         'sv_orig',
    #         {
    #             'multi_V': False,
    #             'init': 'orig'
    #         }
    #     ),
    # ]

    # GridTester('v3_result', 300, pg2).execute()
