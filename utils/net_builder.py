import sys
import numpy as np
import lasagne

##################################################################
# SIMPLE LSTM
##################################################################


def lstm(n=2, D=2, H=100, grad_clip=10):
    '''
    Build a simple lstm architecture.
    n = nombre lstm layers
    input : (N,T,D) input tensor
    output : (N,T,1) output predictions

    with N the number of seq in a batch, T the size of a seq and
    D the number of feature for an input vector

    parameters:
    D : Number of feature for the input layer
    H : Number of hidden units
    GRAD_CLIP : Threeshold value to clip the gradients

    '''
    # input layer (list for variable seq/batch size)
    l_in = lasagne.layers.InputLayer(name='in',
                                     shape=(None, None, D))
    batchsize, seqlen, _ = l_in.input_var.shape
    # lstm layer with tanh non linearity and grad clipper
    l_layer_before = l_in
    for i in range(n):
        l_lstm = lasagne.layers.LSTMLayer(l_layer_before,
                                          H,
                                          name='lstm_%d' % (i),
                                          grad_clipping=grad_clip,
                                          nonlinearity=lasagne.nonlinearities.tanh)
        l_layer_before = l_lstm

    # reshaping prior to feed to the scoringlayer
    l_shp = lasagne.layers.ReshapeLayer(l_lstm, (-1, H))
    # Dense scoring layers
    l_dense = lasagne.layers.DenseLayer(l_shp,
                                        num_units=1,
                                        name='dense',
                                        nonlinearity=lasagne.nonlinearities.rectify)
    # return (N,T) sequence of predictions.
    l_out = lasagne.layers.ReshapeLayer(l_dense, (batchsize, seqlen))

    return l_out
