import os
import sys
import json
import sys
sys.path.append('../')
from utils.Version_1.initialization import *

fognet = os.path.join('~', 'Documents', 'project', 'competition', 'fognet')
conf = {}

####################################
# Model version
conf['version'] = 'version0'

####################################
# Model definition
conf['overwrite'] = True
conf['continue_training'] = False

# Iterator
conf['name'] = 'micro'
conf['feats'] = ['percip_mm', 'humidity', 'temp', 'leafwet450_min',
                 'leafwet460_min', 'leafwet_lwscnt', 'gusts_ms', 'wind_dir', 'wind_ms']
conf['build_ite'] = 'benchmark'
conf['stride'] = 1
conf['batch_size'] = 25
conf['seq_length'] = 200

# pipeline
conf['pipe_list'] = ['MissingValueInputer',
                     'FillRemainingNaN', 'MyStandardScaler']
conf['pipe_kwargs'] = {'MissingValueInputer__method': 'time',
                       'FillRemainingNaN__method': 'bfill'}

# Architecture
conf['type_model'] = 'lstm'
conf['nb_layers'] = 1
conf['which_architecture'] = 'lstm'
conf['grad_clip'] = 1

# Solver
conf['obj_loss_function'] = 'partial_squared_error'
conf['update_rule'] = 'adam'
conf['verbose'] = 11
conf['nb_epochs'] = 1000
conf['patience'] = 25


# Hyperparameters
conf['lr'] = 1e-4
conf['reg'] = 1e-6
conf['hiddens'] = 60

# Initialization
conf['platform'] = get_platform_and_create_folder(fognet)
initialize_work_tree(fognet, conf)
