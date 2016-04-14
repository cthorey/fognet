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
        'pipe': getattr(pipe_def_arima, 'pipe0'),
        'pipe_yield': getattr(pipe_def_arima, 'pipe_yield_base'),
        'parameters_def': {'AR': {'min': 0, 'max': 6, 'step': 1},
                           'D': [0, 1],
                           'MA': {'min': 0, 'max': 6, 'step': 1}},
        'experiment_name': 'SARIMAX/model_0/bbking',
        'description': 'Base model that works best for the moment',
        'verbose': 0}

fognet = os.path.join('~', 'Documents', 'project', 'competition', 'fognet')
_ = build_conf(fognet=fognet, **conf)
