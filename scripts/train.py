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
from utils.train_utils import Model

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

    model = Model(config, mode='train')
    model.train()
