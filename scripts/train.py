################################################################
# libraries dependency

import sys
sys.path.append('..')

import argparse
import os
import sys
import importlib

import json
import theano
import theano.tensor as T
import lasagne
import numpy as np
import matplotlib
import cPickle as pickle
from utils.hook import write_final_score
from utils.data_utils import *
from utils.nolearn_net import NeuralNet
from utils.training_utils import *
from utils.preprocessing import *
from utils.prediction_utils import prediction, make_submission


def train(config):
    ''' The function that allow training giving a specific model given
    a config file '''
    ################################################################
    # Load the preprocessing
    print '\n Loading the prepro pipeline : %s \n' % config['pipe_list']
    pipeline = build_pipeline(config['pipe_list'], config['pipe_kwargs'])

    ################################################################
    # Load the iterator
    # Initialize the batchiterator
    print '\n Loading data iterator using : %s \n' % config['build_ite']
    nb_features, batch_ite_train, batch_ite_val, batch_ite_test, batch_ite_pred = load_data(
        name=config['name'],
        feats=config['feats'],
        build_ite=config['build_ite'],
        pipeline=pipeline)

    ################################################################
    # Build the architecture
    print '\n Build the architecture: %s, %s\n' % (config['model'], config['architecture'])
    model = importlib.import_module(
        'model_defs.%s' % config['model'])
    builder = getattr(model, config['architecture'])
    architecture = builder(D=nb_features, H=config[
        'hiddens'], grad_clip=config['grad_clip'])

    if training:
        ################################################################
        # Model checkpoints
        print '\n Set up the checkpoints\n '
        # Specifc hyperparameters for the name of the checkpoints
        hp = {'lr': config['lr'], 'rg': config['reg'], 'h': config['hiddens']}
        model_fname, save_weights, save_training_history, plot_training_history, early_stopping = initialize_checkpoints(
            config, hp)
        on_epoch_finished = [
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping
        ]

    elif not training:
        on_epoch_finished = []

    ################################################################
    # Initialize solver
    print '\n Initialize the network \n '
    net = NeuralNet(
        layers=architecture,
        regression=True,
        objective_loss_function=getattr(
            lasagne.objectives, config['loss_function']),
        objective_l2=config['reg'],  # L2 regularization
        update=getattr(lasagne.updates, config['update_rule']),
        update_learning_rate=config['lr'],
        batch_iterator_train=batch_ite_train,
        batch_iterator_test=batch_ite_val,
        on_epoch_finished=on_epoch_finished,
        verbose=config['verbose'],
        max_epochs=10000,
    )
    net.initialize()

    ################################################################
    # Reload the weights if we go from an older mode
    if config['continue_training']:
        print 'Loading model params from %s' % config['model_fname']
        net.load_params_from(config['model_fname'])
        with open(config['model_history_fname']) as f:
            net.train_history_ = pickle.load(f)

    ################################################################
    # Fitting
    net.fit(epochs=config['nb_epochs'])

    ################################################################
    # Final score
    print 'Loading best param'
    net.load_params_from(model_fname)

    print 'Evaluating on test set'
    print net.get_score_whole_set(split='test')

    ################################################################
    # Write final score in the folder as a name of txt file
    write_final_score(config, net)

    ################################################################
    # Predict the yield for the whole prediction set
    print 'Run the prediction'
    final_pred = prediction(net, batch_ite_pred)
    make_submission(config, final_pred)

if __name__ == '__main__':
    ################################################################
    # Parse parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, help='Baseline conf file')
    parser.add_argument('--reg', default=1e-7,
                        help='l2 reg parameter', type=float)
    parser.add_argument('--lr', default=1e-4, help='learning rate', type=float)
    parser.add_argument('--hiddens', default=50,
                        help='Number of units', type=int)
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--continue_training', action='store_true')

    # Load the args
    config = vars(parser.parse_args())

    # Deal with loading a previous model or not
    if config['continue_training'] and os.path.exists(config['conf']):
        ################################################################
        # Reload the config file if we continue training
        config.update(parse_conf_file(config['conf']))
        if 'model_name' not in config.keys():
            print('You want to relaod a model which does not exist')
            raise ValueError()
        print 'Loading model from %s' % config['model_fname']
    elif config['continue_training'] and not os.path.exists(config['conf']):
        print('You want to relaod a model which does not exist')
        raise ValueError()
    elif not config['continue_training']:
        # Parse the model_conf file (baseline)
        config.update(parse_conf_file(config['conf']))
        config['time'] = get_current_datetime()

    train(config)
