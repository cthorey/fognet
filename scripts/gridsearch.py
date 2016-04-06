import sys
sys.path.append('..')
import argparse
import os
import sys
import importlib
import itertools
from utils.train_utils import *
from utils.helper import control_type_parameter
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from random import shuffle
from utils import pipe_def
from Oscar import Oscar


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


def train_model_with_oscar(conf, scientist, experiment):

    conf = dict(conf)
    job = scientist.suggest(experiment)
    new_parameters = {key: val for key, val in job.iteritems()}
    parameters = control_type_parameter(new_parameters)
    conf = update_dict(conf, parameters)
    model = Model(config=conf, mode='train', hp=parameters.keys())
    model.train()
    results = {
        'loss': model.get_score_set('val'),
        'rmse_train': model.get_score_set('train'),
        'loss_train': model.get_loss_set('train'),
        'loss_val': model.get_loss_set('val')}
    scientist.update(job, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, help='Baseline conf file')
    parser.add_argument('--oscar', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--continue_training', action='store_true')

    # Load the args
    config = vars(parser.parse_args())
    config.update(parse_conf_file(config['conf']))
    config['time'] = get_current_datetime()

    if config['oscar']:
        parameters_def = {'lr': {'min': 1e-4, 'max': 1e-2},
                          'nb_layers': [2],
                          'reg': {'min': 1e-12, 'max': 1e-6},
                          'stride': {'min': 1, 'max': 5, 'step': 1},
                          'update_rule': ['adam', 'rmsprop'],
                          'hiddens': [30, 50, 100],
                          'seq_length': [150]}
        scientist = Oscar(config['access_token_oscar'])
        experiment = {
            'name': 'lstm_testg_layer_on_top',
            'description': 'just test the hability of oscar to handle my jobs',
            'parameters': parameters_def
        }
        # for i in range(100):
        #     train_model_with_oscar(config, scientist, experiment)
        Parallel(n_jobs=config['nb_cpus'])(delayed(train_model_with_oscar)(
            config, scientist, experiment) for i in range(200))

    else:
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
