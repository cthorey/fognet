import os
import sys
import json
import sys
sys.path.append('..')
from utils.initialization import *
from os.path import expanduser

home = expanduser("~")
fognet = os.path.join(home, 'Documents', 'project', 'competition', 'fognet')
conf = {}

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
conf['pipe_list'] = ['MyImputer', 'MyStandardScaler']
conf['pipe_kwargs'] = {'MyImputer__strategy': 'mean'}

# Architecture
conf['type_model'] = 'lstm'
conf['nb_layers'] = 1
conf['which_architecture'] = 'lstm'
conf['grad_clip'] = 10

# Solver
conf['loss_function'] = 'squared_error'
conf['update_rule'] = 'adam'
conf['verbose'] = 11
conf['nb_epochs'] = 1000
conf['patience'] = 50


# Hyperparameters
conf['lr'] = 1e-4
conf['reg'] = 1e-6
conf['hiddens'] = 60

# Initialization
conf['platform'] = get_platform_and_create_folder(fognet)
initialize_work_tree(fognet, conf)
