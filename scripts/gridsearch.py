import sys
sys.path.append('..')

import argparse
import os
import sys
import importlib
import itertools
from script.train import *


def grid_search(parameters_grid):
    ''' Run a grid search over the parameters '''

    assert all([f in ['lr', 'reg', 'hiddens'] for f in parameters_grid.keys()])

    product = [x for x in apply(itertools.product, parameter_grid.values())]
    runs = [dict(zip(parameter_grid.keys(), p)) for p in product]

    for i, grid in runs:
        config.update(grid)
        train(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, help='Baseline conf file')

    # Load the args
    config = vars(parser.parse_args())
    config.update(parse_conf_file(config['conf']))
    config['time'] = get_current_datetime()

    # grid parameter
    parameters_grid = {'lr': np.logspace(-7, -2, num=10),
                       'reg': np.logspace(-7, -2, num=10),
                       'hidden': range(25, 500, 50)}
    grid_search(config, parameters_grid)
