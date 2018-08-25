#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.datasets import load_dataset
from utils.params import Params
from utils.loggers import Logger


class DualTester:

    def __init__(self, *, root_directory, dataset, mutual, init_params, params, times):

        self.root_directory = root_directory
        self.params = params
        self.init_params = init_params
        self.times = times
        self.dataset = dataset
        self.mutual = mutual

        if root_directory:
            os.makedirs(root_directory, exist_ok=True)

    def target(self, index):

        from main import run, init_uv

        X, C, labels = load_dataset(self.dataset)

        U, V = init_uv(X, C, Params(dict(**self.init_params, **self.mutual)))

        result = {}

        for name, param in self.params.items():
            p = Params({
                **param,
                **self.mutual,
                'initial': (U, V),
                'init': 'preset',
                'C': C
            })

            dest = os.path.join(self.root_directory, name + '.h5.' + str(index)) if self.root_directory else ''

            logger = Logger(dest)
            result = run(X, labels, p, logger)
            logger.log_final(*result)
            logger.close()

        return result

    def execute(self):

        import os
        import multiprocessing.pool as mpp

        pool = mpp.Pool(int(os.cpu_count() * .75))

        pool.map(self.target, range(self.times))


if __name__ == '__main__':

    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('PARAM')

    with open(parser.parse_args().PARAM, 'r') as f:
        params = json.load(f)

    params['root_directory'] = os.path.join(os.path.dirname(__file__), '..', 'results', params['root_directory'])

    DualTester(**params).execute()
