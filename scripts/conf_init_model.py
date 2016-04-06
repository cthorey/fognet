import os
import sys
import json
import sys
sys.path.append('../')
from multiprocessing import cpu_count
from utils.initialization import *
from utils.data_utils import load_raw_data
from utils.helper import myDict
from utils import pipe_def

fognet = os.path.join('~', 'Documents', 'project', 'competition', 'fognet')
conf = {}


####################################
# Version code
conf['version'] = 'version1'
conf['nb_cpus'] = 2

####################################
# Model definition
conf['overwrite'] = True
conf['continue_training'] = False

# Iterator
conf['build_ite'] = 'benchmark'
conf['stride'] = 1
conf['batch_size'] = 25
conf['seq_length'] = 200

# pipeline
# Faire bien attention __ et pas _ pour les parametres
conf['pipe'] = getattr(pipe_def, 'pipe0')

# Architecture
conf['type_model'] = 'lstm'
conf['nb_layers'] = 1
conf['which_architecture'] = 'test_lstm'
conf['grad_clip'] = 1

# Solver
conf['obj_loss_function'] = 'partial_squared_error'
conf['update_rule'] = 'adam'
conf['verbose'] = 11
conf['nb_epochs'] = 10
conf['patience'] = 5


# Hyperparameters
conf['lr'] = 1e-4
conf['reg'] = 1e-6
conf['hiddens'] = 60

# Initialization
conf['platform'], conf[
    'access_token_oscar'] = get_platform_and_create_folder(fognet)
initialize_work_tree(fognet, conf)
