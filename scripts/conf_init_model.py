import os
import sys
import json
import sys
sys.path.append('..')
from utils.training_utils import get_model_name, dump_conf_file
from os.path import expanduser
import platform

home = expanduser("~")

fognet = os.path.join(home, 'Documents', 'project', 'competition', 'fognet')

conf = {}

# Where the model is runed ?
if platform.uname()[1] == 'pss-16.step.univ-paris-diderot.fr':
    conf['platform'] = 'bbking'
    conf['njobs'] = 2
    # laptop sur mon bureau
elif platform.uname()[1] == 'clavius.step.univ-paris-diderot.fr':
    conf['platform'] = 'clavius'
    conf['njobs'] = 8
else:
    raise SystemExit('Platform unknown !')
if not os.path.isdir(os.path.join(fognet, 'models', conf['platform'])):
    os.mkdir(os.path.join(fognet, 'models', conf['platform']))

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
conf['patience'] = 25

path_base_model = os.path.join(
    fognet, 'models', conf['platform'], conf['model'])
if not os.path.isdir(path_base_model):
    os.mkdir(path_base_model)

dir_new_model = get_model_name(path_base_model)
try:
    'Initialize the model tree'
    os.mkdir(dir_new_model)
except:
    raise ValueError(
        'Cannot create the directory for the model %s' % (dir_new_model))


conf['root'] = dir_new_model
dump_conf_file(conf, dir_new_model)
