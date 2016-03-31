import sys
sys.path.append('..')
import argparse
import os
import sys
import importlib
import itertools
from utils.train_utils import *
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from random import shuffle
from utils import pipe_def


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

    product = [x for x in apply(itertools.product, parameters_grid.values())]
    conf_runs = [dict(zip(parameters_grid.keys(), p)) for p in product]
    confs = map(lambda d: update_dict(config, d), conf_runs)
    shuffle(confs)
    return confs


def train_model(config, hp):
    model = Model(config=config, mode='train', hp=hp)
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, help='Baseline conf file')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--continue_training', action='store_true')

    # Load the args
    config = vars(parser.parse_args())
    config.update(parse_conf_file(config['conf']))
    config['time'] = get_current_datetime()

    parameters_grid = {'lr': [1e-2, 1e-3, 1e-4],
                       'nb_layers': [1, 2, 3],
                       'reg': [1e-4, 1e-6, 0.0],
                       'stride': [1, 2],
                       'update_rule': ['adam', 'rmsprop'],
                       'hiddens': [20, 50, 100, 200],
                       'seq_length': [100, 200, 300]}

    confs = conf_generator(config, parameters_grid)
    print('We are going to run %d different models' % (len(confs)))
    Parallel(n_jobs=config['nb_cpus'])(delayed(train_model)(
        conf, parameters_grid.keys()) for conf in confs)
