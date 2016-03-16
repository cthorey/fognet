import sys
sys.path.append('..')

import numpy as np
import lasagne
from utils.nolearn_net import NeuralNet
from nolearn.lasagne.handlers import SaveWeights

from utils.hooks import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping
)

##################################################################
# Model checkpoints
##################################################################

model_fname = './models/lstm_16_03.pkl'
model_history_fname = './models/lstm_16_03_history.pkl'
model_graph_fname = './models/lstm_16_03_history.png'

save_weights = SaveWeights(model_fname, only_best=True, pickle=False)
save_training_history = SaveTrainingHistory(model_history_fname)
plot_training_history = PlotTrainingHistory(model_graph_fname)
early_stopping = EarlyStopping(patience=150)


##################################################################
# Model definition
##################################################################

######################
# Model parameters

D = 2
H = 100
GRAD_CLIP = 10


######################
# Build the model, layer by layer

# First, we build the network, starting with an input layer
# Recurrent layers expect input of shape
# (N, T, D) with
# N the number of example in the batch
# T the size of the sequence
# D the nb of considered features

l_in = lasagne.layers.InputLayer(name='in',
                                 shape=(None, None, D))
batchsize, seqlen, _ = l_in.input_var.shape


# We now build the LSTM layer which takes l_in as the input layer
# We clip the gradients at GRAD_CLIP to prevent the problem of exploding
# gradients.
l_lstm = lasagne.layers.LSTMLayer(l_in,
                                  H,
                                  name='lstm',
                                  grad_clipping=GRAD_CLIP,
                                  nonlinearity=lasagne.nonlinearities.tanh)

# Reshaping prior to feed to the dense layer
l_shp = lasagne.layers.ReshapeLayer(l_lstm, (-1, H))
l_dense = lasagne.layers.DenseLayer(l_shp,
                                    num_units=1,
                                    name='dense',
                                    nonlinearity=lasagne.nonlinearities.identity)
# Return the adequate shape (N,T)
l_out = lasagne.layers.ReshapeLayer(l_dense, (batchsize, seqlen))


net = NeuralNet(
    layers=l,

    regression=True,
    use_label_encoder=False,

    objective_loss_function=nn.objectives.squared_error,
    objective_l2=1e-7,

    update=nn.updates.adam,
    update_learning_rate=1e-4,

    train_split=TrainSplit(0.15, stratify=False, random_state=42),
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history,
        early_stopping
    ],

    verbose=10,

    max_epochs=1000,
)
