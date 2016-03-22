import sys
sys.path.append('..')
from joblib import Parallel, delayed
import argparse
import os
import sys
import importlib
import itertools
from train import *


def conf_generator(config, parameters_grids):

    assert all([f in ['lr', 'reg', 'hiddens'] for f in parameters_grid.keys()])
    product = [x for x in apply(itertools.product, parameters_grid.values())]
    conf_runs = [dict(zip(parameters_grid.keys(), p)) for p in product]
    for conf_run in conf_runs:
        config.update(conf_run)
        yield config


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
    confs = conf_generator(config, parameters_grid)
    out = Parallel(n_jobs=config['njobs'], verbose=100)(delayed(train)(conf)
                                                        for conf in confs)
