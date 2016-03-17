import os
import sys
import json
import sys
sys.path.append('..')
from utils.training_utils import get_model_name
from os.path import expanduser
home = expanduser("~")

fognet = os.path.join(home, 'Documents', 'project', 'competition', 'fognet')

conf = {}

####################################
# Model definition
# Iterator
conf['processing'] = 'benchmark'

# Architecture
conf['model'] = 'lstm'
conf['architecture'] = 'build_simple_lstm'
conf['grad_clip'] = 10

# Solver
conf['loss_function'] = 'squared_error'
conf['update_rule'] = 'adam'
conf['verbose'] = 11
conf['nb_epochs'] = 1
conf['patience'] = 150

dir_new_model = get_model_name(os.path.join(fognet, 'models', conf['model']))
try:
    'Initialize the model tree'
    os.mkdir(dir_new_model)
except:
    raise ValueError(
        'Cannot create the directory for the model %s' % (dir_new_model))

conf['root'] = dir_new_model
with open(os.path.join(dir_new_model, 'conf_model.json'), 'w+') as f:
    json.dump(conf,
              f,
              sort_keys=True,
              indent=4,
              ensure_ascii=False)
