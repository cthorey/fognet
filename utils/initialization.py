import platform
import os
import json
from helper import *
import pipe_def_arima


def get_platform_and_create_folder(fognet):
    ''' Used in conf_init_model '''
    # Where the model is runed ?
    if platform.uname()[1] == 'pss-16.step.univ-paris-diderot.fr':
        name_platform = 'bbking'
        # laptop ur mon bureau
    elif platform.uname()[1] == 'clavius.step.univ-paris-diderot.fr':
        name_platform = 'clavius'
    elif platform.uname()[1] in ['Thoreys-MBP', 'Thoreys-MacBook-Pro.local']:
        name_platform = 'ray'
    else:
        raise SystemExit('Platform unknown !')
    path_models = os.path.expanduser(
        os.path.join(fognet, 'models', name_platform))
    if not os.path.isdir(path_models):
        os.mkdir(path_models)
    access_token_oscar = 'bv46kOZd1WwBFo33lzTZjFTKKUOTGgA6RwlZQ6CWag9zfnNlbnNvdXQtb3NjYXJyEQsSBFVzZXIYgICAgN7PjQoM'
    return name_platform, access_token_oscar


def initialize_work_tree(fognet, conf):
    #########################
    ##
    path_base_model = os.path.join(
        fognet, 'models', conf['platform'], conf['type_model'])
    print os.path.expanduser(path_base_model)
    if not os.path.isdir(os.path.expanduser(path_base_model)):
        os.mkdir(os.path.expanduser(path_base_model))

    dir_new_model = get_model_name(os.path.expanduser(path_base_model))
    path_dir_new_model = os.path.join(path_base_model, dir_new_model)
    try:
        'Initialize the model tree'
        os.mkdir(os.path.expanduser(path_dir_new_model))
    except:
        raise ValueError(
            'Cannot create the directory for the model %s' % (os.path.expanduser(path_dir_new_model)))

    conf['root'] = path_dir_new_model
    dump_conf_file(conf, os.path.expanduser(path_dir_new_model))
    print('Conf file put in %s' % (os.path.expanduser(path_dir_new_model)))


def get_model_name(path):
    ''' Given a directory where you want to run a new bunch of  model
    define by the baseline in the conf_init file, automatically select
    the name of the model by incrementing by 1 the largest previous
    model in the name '''
    existing_models = [f for f in os.listdir(
        path) if f.split('_')[0] == 'model']
    if len(existing_models) == 0:
        model = -1
    else:
        model = max([int(f.split('_')[1]) for f in existing_models])
    return 'model_' + str(model + 1)

numerical_feature = ['gusts_ms', 'humidity', 'leafwet450_min', 'leafwet460_min', 'leafwet_lwscnt',
                     'percip_mm', 'temp', 'wind_dir', 'wind_ms', 'guel_T', 'guel_P0', 'guel_P',
                     'guel_U', 'guel_Ff', 'guel_ff10', 'guel_Td', 'sidi_T', 'sidi_Po', 'sidi_P',
                     'sidi_Pa', 'sidi_U', 'sidi_Ff', 'sidi_Tn', 'sidi_Tx', 'sidi_VV',
                     'sidi_Td', 'sidi_tR', 'sidi_Tg', 'sidi_sss', 'aga_T',
                     'aga_P0', 'aga_P', 'aga_U', 'aga_Ff', 'aga_ff10', 'aga_Td']

base_conf = {'nb_cpus': 2,
             'overwrite': True,
             'continue_training': False,
             'type_model': 'arima',
             'which_architecture': 'SARIMAX',
             'pca_components': 0,
             'AR': 0,
             'MA': 1,
             'D': 1,
             'Season_AR': 0,
             'Season_MA': 0,
             'Season_D': 0,
             'Season_Period': 0,
             'nb_folds': 10,
             'size_gap': 96,
             'features_base': ['leafwet450_min', 'leafwet_lwscnt', 'temp',
                               'gusts_ms', 'leafwet460_min', 'humidity'],
             'numerical_feature': numerical_feature,
             'num_features_extra': 1,
             'inputer': 'InterpolateMissingValueInputer',
             'num_lags_regressors': 2,
             'seasonal_inter_lags': 6,
             'pipe_yield': getattr(pipe_def_arima, 'pipe_yield_base'),
             'experiment_name': 'SARIMAX/model_4/clavius',
             'description': 'Explore the parameters alone of the distribution. ' +
             'What is the best combination for the order parameters' +
             'of the SARIMAX model-Seasonal effect. Data macro',
             'parameters_def': {'AR': {'min': 0, 'max': 8, 'step': 1},
                                'D': [0, 1, 2],
                                'MA': {'min': 0, 'max': 8, 'step': 1},
                                'Season_AR': {'min': 0, 'max': 12, 'step': 1},
                                'Season_D': [0, 1],
                                'Season_MA': {'min': 0, 'max': 12, 'step': 1},
                                'Season_Period': [1, 6, 12]},
             'parameters_grid': {'AR': range(7),
                                 'D': [0, 1],
                                 'MA': range(7)},
             'verbose': 0}


def build_conf(fognet, **kwargs):
    conf = base_conf
    conf.update(kwargs)
    # Initialization
    conf['platform'], conf[
        'access_token_oscar'] = get_platform_and_create_folder(fognet)
    initialize_work_tree(fognet, conf)
    return conf
