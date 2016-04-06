import sys
sys.path.append('../../')
import argparse
import os
import sys
import importlib
import itertools
from utils.arima_utils import *
from utils.helper import *
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
    parameters = control_type_parameter_arima(new_parameters)
    conf = update_dict(conf, parameters)
    model = ArimaModel(config=conf, mode='train', hp=parameters.keys())
    model.fit()
    resuts_key = ['rmse', 'aic', 'bic', 'hqic']
    results = {'loss': model.val_rmse}
    results.update({'train_%s' % (key): getattr(
        model, 'train_%s' % (key)) for key in results})
    results.update({'test_%s' % (key): getattr(
        model, 'test_%s' % (key)) for key in results})
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
        parameters_def = {'AR': {'min': 0, 'max': 12, 'step': 1},
                          'D': [0, 1],
                          'MA': {'min': 0, 'max': 12, 'step': 1},
                          'Season_RA': {'min': 0, 'max': 12, 'step': 1},
                          'Season_D': [0, 1],
                          'Season_MA': {'min': 0, 'max': 12, 'step': 1},
                          'Season_Period': [12, 24, 48]}
        scientist = Oscar(config['access_token_oscar'])
        experiment = {
            'name': 'SARIMAX - Model selection',
            'description': 'What about that',
            'parameters': parameters_def
        }
        # Parallel(n_jobs=config['nb_cpus'])(delayed(train_model_with_oscar)(
        #     config, scientist, experiment) for i in range(200))
        while True:
            train_model_with_oscar(config, scientist, experiment)

    else:
        parameters_grid = {'RA': [1, 2, 3]}
        confs = conf_generator(config, parameters_grid)
        print('We are going to run %d different models' % (len(confs)))
        Parallel(n_jobs=config['nb_cpus'])(delayed(train_model)(
            conf, parameters_grid.keys()) for conf in confs)
