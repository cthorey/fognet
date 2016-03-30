import os
import sys
import json
import sys
sys.path.append('../')
from utils.initialization import *
from utils.data_utils import load_raw_data
from utils.helper import myDict

fognet = os.path.join('~', 'Documents', 'project', 'competition', 'fognet')
conf = {}


####################################
# Version code
conf['version'] = 'version1'

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
# Faire bien attention __ et pas _ pour les parametres
data = load_raw_data()
pipe_list_micro = ['FeatureSelector',
                   'MissingValueInputer',
                   'FillRemainingNaN',
                   'MyStandardScaler']
pipe_list_macro = ['FeatureSelector',
                   'NumericFeatureSelector',
                   'MissingValueInputer',
                   'FillRemainingNaN',
                   'MyStandardScaler']

base_kwargs = {'MissingValueInputer__method': 'time',
               'FillRemainingNaN__method': 'bfill'}
kwargs_micro = base_kwargs.copy()
kwargs_micro.update({'FeatureSelector__features': data['micro_feats'].keys()})

kwargs_macro_aga = base_kwargs.copy()
kwargs_macro_aga .update(
    {'FeatureSelector__features': data['aga_feats'].keys()})

# A cause du resampling, pas toutes les features dans sidi
sidi_feats = ['T', 'Po', 'P', 'Pa', 'U', 'Ff',
              'Tn', 'Tx', 'VV', 'Td', 'tR', 'Tg', 'sss']
sidi_feats = ['sidi_%s' % (f) for f in sidi_feats]
kwargs_macro_sidi = base_kwargs.copy()
kwargs_macro_sidi.update(
    {'FeatureSelector__features': sidi_feats})

kwargs_macro_guel = base_kwargs.copy()
kwargs_macro_guel.update(
    {'FeatureSelector__features': data['guel_feats'].keys()})

conf['pipe_list'] = {'micro': pipe_list_micro,
                     'macro_aga': pipe_list_macro,
                     'macro_sidi': pipe_list_macro,
                     'macro_guel': pipe_list_macro}
conf['pipe_kwargs'] = {'micro': kwargs_micro,
                       'macro_aga': kwargs_macro_aga,
                       'macro_sidi': kwargs_macro_sidi,
                       'macro_guel': kwargs_macro_guel
                       }
assert conf['pipe_list'].keys() == conf['pipe_kwargs'].keys()


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
