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
