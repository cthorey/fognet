import sys
sys.path.append('..')
import argparse
import os
import sys
import importlib
import itertools
from train import *
from multiprocessing import Pool
from multiprocessing import cpu_count
from joblib import Parallel, delayed


def update_dict(config, new_parameters):
    ''' return an update dict based on config
    input:
    config : baseline dict
    new_parameters : new parameter dict
    '''
    d = dict(config)  # Creat new dict, baseline=config
    d.update(new_parameters)  # update the new dict with the new parameters
    return d  # return it ! Trust me , only way to get it done


def conf_generator(config, parameters_grids):

    assert all([f in ['lr', 'reg', 'hiddens'] for f in parameters_grid.keys()])
    product = [x for x in apply(itertools.product, parameters_grid.values())]
    conf_runs = [dict(zip(parameters_grid.keys(), p)) for p in product]
    confs = map(lambda d: update_dict(config, d), conf_runs)
    return confs


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
    parameters_grid = {'lr': np.logspace(-7, 0, num=10),
                       'reg': np.logspace(-7, -3, num=10),
                       'hiddens': range(10, 250, 25)}
    confs = conf_generator(config, parameters_grid)
    print('We are going to run %d different models' % (len(confs)))
    Parallel(n_jobs=cpu_count())(delayed(train)(conf) for conf in confs)
