import os
import json
from nolearn.lasagne.handlers import SaveWeights
from hook import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping
)
from time import strftime


def check_folder(config, folder):
    output_exists = os.path.isdir(folder)
    if output_exists and not config['overwrite']:
        print 'Model output exists. Use --overwrite'
        sys.exit(1)
    elif not output_exists:
        os.mkdir(folder)


def initialize_checkpoints(config, hp):
    # Model checkpoints
    name = '_'.join([f + '_' + str(g) for f, g in hp.iteritems()])
    fname = os.path.join(config['root'], name)
    check_folder(config, fname)
    model_fname = os.path.join(fname, 'model.pkl')
    model_history_fname = os.path.join(fname, 'model_history.pkl')
    model_graph_fname = os.path.join(fname, 'model_history.png')

    save_weights = SaveWeights(model_fname, only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory(model_history_fname)
    plot_training_history = PlotTrainingHistory(model_graph_fname)
    early_stopping = EarlyStopping(patience=config['patience'])

    # Add some useful key to the config file
    config['folder'] = fname
    config['model_name'] = name
    config['model_fname'] = model_fname
    config['model_history_fname'] = model_history_fname
    config['model_graph_fname'] = model_graph_fname

    dump_conf_file(config, fname)

    return model_fname, save_weights, save_training_history, plot_training_history, early_stopping
