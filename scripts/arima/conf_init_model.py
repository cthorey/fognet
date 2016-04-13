import os
import sys
import json
import sys
sys.path.append('../../')
from multiprocessing import cpu_count
from utils.initialization import *
from utils.data_utils import load_raw_data
from utils.helper import myDict
from utils import pipe_def_arima


conf = {'nb_cpus': 2,
        'type_model': 'arima',
        'which_architecture': 'SARIMAX',
        'pca_components': 0,
        'AR': 0,
        'MA': 1,
        'D': 1,
        'Season_AR': 0,
        'Season_MA': 0,
        'Season_D': 0,
        'Season_Period': 0,
        'pipe': getattr(pipe_def_arima, 'pipe1'),
        'pipe_yield': getattr(pipe_def_arima, 'pipe_yield'),
        'parameters_def': {'AR': {'min': 0, 'max': 8, 'step': 1},
                           'D': [0, 1, 2],
                           'MA': {'min': 0, 'max': 8, 'step': 1},
                           'Season_AR': {'min': 0, 'max': 12, 'step': 1},
                           'Season_D': [0, 1],
                           'Season_MA': {'min': 0, 'max': 12, 'step': 1},
                           'Season_Period': [1, 6, 12]},
        'experiment_name': 'SARIMAX/model_4/clavius',
        'description': 'Explore the parameters alone of the distribution. ' +
        'What is the best combination for the order parameters' +
        'of the SARIMAX model-Seasonal effect. Data macro',
        'parameters_grid': {'AR': range(7),
                            'D': [0, 1],
                            'MA': range(7)},
        'verbose': 0}

fognet = os.path.join('~', 'Documents', 'project', 'competition', 'fognet')
_ = build_conf(fognet=fognet, **conf)
