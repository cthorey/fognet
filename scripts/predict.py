import sys
sys.path.append('..')
import argparse
import importlib
from time import strftime
import numpy as np
import pandas as pd
import cPickle as pickle
from utils.training_utils import *
from utils.data_utils import *
from utils.nolearn_net import NeuralNet
from utils.prediction_utils import prediction
from utils.preprocessing import *
import json
import theano
import theano.tensor as T
import lasagne

################################################################
# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('--conf', required=True, help='Conf file for model')

################################################################
# Get the parameters from the file
# get the path where the conf file for the model is stored
config = vars(parser.parse_args())
# get the model parameters and store them in a ditct
config.update(parse_conf_file(config['conf']))


################################################################
# Output file to store the submission
output_fname = os.path.join(
    config['folder'], 'submissions_%s.csv' % get_current_datetime())
print 'Will write output to %s' % output_fname

################################################################
# Load the preprocessing
print '\n Loading the prepro pipeline : %s \n' % config['pipe_list']
pipeline = build_pipeline(config['pipe_list'], config['pipe_kwargs'])

################################################################
# Load the iterator for prediction
print '\n Loading data iterator using : %s \n' % config['build_ite']
nb_features, _, _, _, batch_ite_pred = load_data(
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

################################################################
# Initialize  the model
print '\n Initialize the network \n '
net = NeuralNet(
    layers=architecture,
    regression=True,
    objective_loss_function=getattr(
        lasagne.objectives, config['loss_function']),
    objective_l2=config['reg'],  # L2 regularization
    update=getattr(lasagne.updates, config['update_rule']),
    update_learning_rate=config['lr'],
    verbose=config['verbose'],
    max_epochs=10000,
)
net.initialize()

################################################################
# Load the model
print 'Loading model weights from %s' % config['model_fname']
net.load_weights_from(config['model_fname'])

################################################################
# Predict the yield for the whole prediction set
print 'Run the prediction'
final_pred = prediction(net, batch_ite_pred)

################################################################
# Merge and produce  the submission file
submission_df = load_raw_data()['submission_format']
final_pred_format = submission_df.join(final_pred, how='left')
submission_df['yield'] = final_pred_format['yield_pred']

################################################################
# Remove value below zero !
submission_df[submission_df['yield'] < 0.0] = 0

################################################################
# Store to a txt file
submission_df.to_csv(output_fname)
