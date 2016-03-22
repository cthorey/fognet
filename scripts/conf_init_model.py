import os
import sys
import json
import sys
sys.path.append('..')
from utils.training_utils import get_model_name, dump_conf_file
from os.path import expanduser
home = expanduser("~")

fognet = os.path.join(home, 'Documents', 'project', 'competition', 'fognet')

conf = {}

####################################
# Model definition
# Iterator
conf['name'] = 'micro'
conf['feats'] = ['percip_mm', 'humidity', 'temp', 'leafwet450_min',
                 'leafwet460_min', 'leafwet_lwscnt', 'gusts_ms', 'wind_dir', 'wind_ms']
conf['build_ite'] = 'benchmark'
conf['pipe_list'] = ['MyImputer', 'MyStandardScaler']
conf['pipe_kwargs'] = {'MyImputer__strategy': 'mean'}

# Architecture
conf['model'] = 'lstm'
conf['architecture'] = 'build_simple_lstm'
conf['grad_clip'] = 50

# Solver
conf['loss_function'] = 'squared_error'
conf['update_rule'] = 'adam'
conf['verbose'] = 11
conf['nb_epochs'] = 1000
conf['patience'] = 15

dir_new_model = get_model_name(os.path.join(fognet, 'models', conf['model']))
try:
    'Initialize the model tree'
    os.mkdir(dir_new_model)
except:
    raise ValueError(
        'Cannot create the directory for the model %s' % (dir_new_model))


conf['root'] = dir_new_model
dump_conf_file(conf, dir_new_model)
