import platform
import os
import json
from helper import *


def get_platform_and_create_folder(fognet):
    ''' Used in conf_init_model '''
    # Where the model is runed ?
    if platform.uname()[1] == 'pss-16.step.univ-paris-diderot.fr':
        name_platform = 'bbking'
        # laptop ur mon bureau
    elif platform.uname()[1] == 'clavius.step.univ-paris-diderot.fr':
        name_platform = 'clavius'
    elif platform.uname()[1] == 'Thoreys-MBP':
        name_platform = 'ray'
    else:
        raise SystemExit('Platform unknown !')
    if not os.path.isdir(os.path.join(fognet, 'models', name_platform)):
        os.mkdir(os.path.join(fognet, 'models', name_platform))

    return name_platform


def initialize_work_tree(fognet, conf):
    #########################
    ##
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
    print('Conf file put in %s' % (dir_new_model))


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
    return os.path.join(path, 'model_' + str(model + 1))
