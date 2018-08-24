#!/usr/bin/env python3

import os
import json
import shutil
from tester import DualTester
from itertools import product


class GridSearcher:

    def __init__(self, params):

        scales = params.pop('scales')

        grids = []
        for param_name, param_scale in scales.items():
            grids.append([
                (param_name, param_value)
                for param_value in param_scale
            ])

        params_ = {}
        for index, param_tuple in enumerate(product(*grids)):
            params_[str(index)] = dict(param_tuple)

        params.update({
            'params': params_,
        })

        params.setdefault('times', 1)

        print(params)
        self.params = params

        root_directory = params.get('root_directory', None)

        if root_directory:
            shutil.rmtree(root_directory, True)
            os.makedirs(root_directory, exist_ok=True)

            with open(os.path.join(root_directory, 'params.json'), 'w') as f:
                json.dump(params_, f)

    def execute(self):

        DualTester(**params).execute()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('PARAM')

    with open(parser.parse_args().PARAM, 'r') as f:
        params = json.load(f)

    params['root_directory'] = os.path.join(os.path.dirname(__file__), '..', 'results', params['root_directory'])

    GridSearcher(params).execute()
