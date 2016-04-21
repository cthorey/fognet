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

#########################
# Definition

# Description
experiment_name = 'SARIMAX/model_7/bbking'
description = 'Run experiments'

# Oscar grid
parameters_def = {'AR': {'min': 0, 'max': 6, 'step': 1},
                  'D': [0, 1],
                  'MA': {'min': 0, 'max': 6, 'step': 1},
                  'num_features_extra': {'min': 0, 'max': 10, 'step': 1},
                  'inputer': ['InterpolateMissingValueInputer',
                              'EWMAMissingValueInputer',
                              'AutoArimaInputer'],
                  'num_lags_regressors': {'min': 1, 'max': 6, 'step': 1},
                  'seasonal_inter_lags': [1, 12, 24]}

# 'Season_AR': {'min': 0, 'max': 12, 'step': 1},
# 'Season_D': [0, 1],
# 'Season_MA': {'min': 0, 'max': 12, 'step': 1},
# 'Season_Period': [1, 6, 12]}
# Brut grid
parameters_grid = {'AR': range(7),
                   'D': [0, 1],
                   'MA': range(7),
                   'num_features_extra': range(0, 10),
                   'inputer': ['InterpolateMissingValueInputer',
                               'EWMAMissingValueInputer',
                               'AutoArimaInputer'],
                   'num_lags_regressors': range(1, 8),
                   'seasonal_inter_lags': [1, 12, 24],
                   }


#########################
# main

conf = {'parameters_def': parameters_def,
        'parameters_grid': parameters_grid,
        'experiment_name': experiment_name,
        'description': description}


fognet = os.path.join('~', 'Documents', 'project', 'competition', 'fognet')
_ = build_conf(fognet=fognet, **conf)
