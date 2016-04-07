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

fognet = os.path.join('~', 'Documents', 'project', 'competition', 'fognet')
conf = {}


####################################
# Version code
conf['nb_cpus'] = 2

####################################
# Model definition
conf['overwrite'] = True
conf['continue_training'] = False

# pipeline
# Faire bien attention __ et pas _ pour les parametres
conf['pipe'] = getattr(pipe_def_arima, 'pipe0')
conf['pipe_yield'] = getattr(pipe_def_arima, 'pipe_yield')

# Architecture
conf['type_model'] = 'arima'
conf['which_architecture'] = 'SARIMAX'

# Hyperparameters
conf['pca_components'] = 0
conf['AR'] = 0
conf['MA'] = 1
conf['D'] = 1
conf['Season_AR'] = 0
conf['Season_MA'] = 0
conf['Season_D'] = 0
conf['Season_Period'] = 0

# Oscar stuff
conf['parameters_def'] = {'AR': {'min': 0, 'max': 12, 'step': 1},
                          'D': [0, 1, 2],
                          'MA': {'min': 0, 'max': 12, 'step': 1}
                          }
# 'Season_AR': {'min': 0, 'max': 3, 'step': 1},
# 'Season_D': [0, 1],
# 'Season_MA': {'min': 0, 'max': 3, 'step': 1},
# 'Season_Period': [1, 3]

conf['experiment_name'] = 'ARIMAX - model_0 clavius - explore the orders parameters'
conf[
    'description'] = ' Test ARIMAX with the micro datasets'

conf['verbose'] = 0
# Initialization
conf['platform'], conf[
    'access_token_oscar'] = get_platform_and_create_folder(fognet)


initialize_work_tree(fognet, conf)
