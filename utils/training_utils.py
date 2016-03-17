import sys
sys.path.append('..')
import os
import json
from nolearn.lasagne.handlers import SaveWeights
from utils.hook import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping
)


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


def check_folder(args, folder):
    output_exists = os.path.isdir(folder)
    if output_exists and not args.overwrite:
        print 'Model output exists. Use --overwrite'
        sys.exit(1)
    elif not output_exists:
        os.mkdir(folder)


def parse_conf_file(conf_file):
    ''' parse the configuration file'''

    with open(conf_file, 'r') as f:
        conf = json.load(f)
    return conf


def initialize_checkpoints(args, config, hp):
    # Model checkpoints
    name = '_'.join([f + '_' + str(g) for f, g in hp.iteritems()])
    fname = os.path.join(config['root'], name)
    check_folder(args, fname)

    model_fname = os.path.join(fname, 'model.pkl')
    model_history_fname = os.path.join(fname, 'model_history.pkl')
    model_graph_fname = os.path.join(fname, 'model_history.png')

    save_weights = SaveWeights(model_fname, only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory(model_history_fname)
    plot_training_history = PlotTrainingHistory(model_graph_fname)
    early_stopping = EarlyStopping(patience=config['patience'])

    return model_fname, save_weights, save_training_history, plot_training_history, early_stopping
