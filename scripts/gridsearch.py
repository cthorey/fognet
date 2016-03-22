import sys
sys.path.append('..')

import argparse
import os
import sys
import importlib
import itertools
from train import *


def grid_search(config, parameters_grid):
    ''' Run a grid search over the parameters

    config : dictionary to run the lstm
    parameters_grid: a parameter grid with different value over which
    you want the model to run.
    '''

    assert all([f in ['lr', 'reg', 'hiddens'] for f in parameters_grid.keys()])

    product = [x for x in apply(itertools.product, parameters_grid.values())]
    runs = [dict(zip(parameters_grid.keys(), p)) for p in product]

    print('%d model are about to be run' % (len(runs)))
    for i, grid in enumerate(runs):
        print 'Still %d model to be run' % (len(runs) - i)
        config.update(grid)
        train(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, help='Baseline conf file')
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--continue_training', action='store_true')

    # Load the args
    config = vars(parser.parse_args())
    config.update(parse_conf_file(config['conf']))
    config['time'] = get_current_datetime()

    # grid parameter
    parameters_grid = {'lr': np.logspace(-3, -2, num=2),
                       'reg': [1e-7],
                       'hiddens': [25, 50]}
    grid_search(config, parameters_grid)
