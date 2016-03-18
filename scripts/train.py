################################################################
# libraries dependency

import sys
sys.path.append('..')

import argparse
import os
import sys
import importlib
from time import strftime

import json
import theano
import theano.tensor as T
import lasagne
import numpy as np
import matplotlib
import cPickle as pickle
from utils.data_utils import *
from utils.nolearn_net import NeuralNet
from utils.training_utils import *

################################################################
# Parse parameters

parser = argparse.ArgumentParser()
parser.add_argument('--conf', required=True, help='Baseline conf file')
parser.add_argument('--reg', default=1e-7, help='l2 reg parameter')
parser.add_argument('--lr', default=1e-4, help='learning rate')
parser.add_argument('--hiddens', default=150, help='Number of units')
parser.add_argument('--overwrite', action='store_true', default=True)
parser.add_argument('--continue_training', action='store_true')

# Load the args
config = vars(parser.parse_args())
# Parse the model_conf file (baseline)
config.update(parse_conf_file(config['conf']))

################################################################
# Load the iterator
# Initialize the batchiterator
print '\n Loading data iterator using : %s \n' % config['processing']
nb_features, batch_ite_train, batch_ite_val, batch_ite_pred = load_data(
    name=config['name'], feats=config['feats'], processing=config['processing'])

################################################################
# Build the architecture
print '\n Build the architecture: %s, %s\n' % (config['model'], config['architecture'])
model = importlib.import_module(
    'model_defs.%s' % config['model'])
builder = getattr(model, config['architecture'])
architecture = builder(D=nb_features, H=config[
                       'hiddens'], grad_clip=config['grad_clip'])

################################################################
# Model checkpoints
print '\n Set up the checkpoints\n '
# Specifc hyperparameters for the name of the checkpoints
hp = {'lr': config['lr'], 'rg': config['reg'], 'h': config['hiddens']}
model_fname, save_weights, save_training_history, plot_training_history, early_stopping = initialize_checkpoints(
    config, hp)

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
    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history,
        early_stopping
    ],
    verbose=config['verbose'],
    max_epochs=10000,
)
net.initialize()

################################################################
# Reload the weights if we go from an older mode
if config['continue_training'] and os.path.exists(model.model_fname):
    print 'Loading model params from %s' % model.model_fname
    net.load_params_from(model.model_fname)
    with open(model.model_history_fname) as f:
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
