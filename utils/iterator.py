from __future__ import print_function

import numpy as np
import pandas as pd


class BaseBatchIterator(object):

    '''
    Class to proceed from the dataframe to batch of data to be
    feed in the rnn

    '''

    def __init__(self, feats=[], label='yield', batch_size=25, size_seq=25, stride=1, verbose=False):
        # feature to take from the dataframe
        self.feats = feats
        self.nfeats = len(feats)
        # Target
        self.labels = label
        # Size of the sequence you want to enroll in the rnn
        self.seq_size = size_seq
        # Stride for the construction of the sequence.
        # For instance, if we have [xa,xb,xc,xd,xe,xf] where each x is a D-dimensional vector.
        # A stride of 2 with a size_seq of 4, willr etuyrn [[xa,xb,xc,xd],[xc,xd,xe,xf]]
        # Looks like a conv 1D !
        self.stride = stride
        # Size of the batch to process each step of the gradient update
        self.batch_size = batch_size
        self.verbose = False

    def __call__(self, df):
        self.df = df
        self.stack_seqs = self.stack_sequence()
        return self

    def stack_sequence(self):
        ''' 
        This function allows to take the training dataframe, and 
        return a dict of (X,y) tupple. Each key correspond to a 
        different group - define above. Each group is consituted by a 
        continuous sequence of observation (2H lags). 
        For each group, we input a dataframe of size (N,D). N obs/ D features.
        We then build sequence of observation from it. A trivial 
        way to do it is to take df.iloc[0:seq_size],df.iloc[seq_size:2seq_size]
        and so on. 

        However, we proceed as for a 1D convolution. For instance, 
        if we have [xa,xb,xc,xd,xe,xf] where each x is a D-dimensional vector.
        The X, for a stride of 2 with a size_seq of 4, should be [[xa,xb,xc,xd],[xc,xd,xe,xf]].
        The number of sequence is given by the simple formula 
        n_sequences = (N-seq_size)/stride +1

        '''
        # The input dataframe is basically (N,D) dimensional tensor.
        # N is the number of obs
        # D is the number of feature to consider !

        # Get the groups of continuous obs from the df
        gps = self.df.groupby('group')
        n_groups = gps.ngroups  # Nb groups, define by timelag between obs
        # Initialize a dict to store (X,y)
        stack_seqs = dict.fromkeys(set(self.df.group))
        for key, gp in gps:
            nb_obs = gp.shape[0]  # Nb of obs in the group
            # Nb of sequences
            nb_seqs = (nb_obs - self.seq_size) / self.stride + 1

            # Begin of the processing
            for k in range(nb_seqs):
                kmin = k * self.stride  # lower bound window
                kmax = k * self.stride + self.seq_size  # Upper boud window
                X_tmp = np.array(gp[self.feats].iloc[kmin:kmax])[
                    np.newaxis, :]  # Get the X
                y_tmp = np.array(gp[self.labels].iloc[kmin:kmax])[
                    np.newaxis, :]  # Get the y
                if k == 0:
                    X = X_tmp
                    y = y_tmp
                else:
                    X = np.vstack((X, X_tmp))  # Stack them
                    y = np.vstack((y, y_tmp))  # Stack them
            # At the end, X has a shape (N,T,D)
            # y has a shape (N,T)
            stack_seqs[key] = (X, y)
        return stack_seqs

    def __iter__(self):
        # Iterator
        n_groups = len(self.stack_seqs.keys())  # Nb groups
        for gp, (X, y) in self.stack_seqs.iteritems():
            n_samples = X.shape[0]
            bs = self.batch_size
            n_batches = (n_samples + bs - 1) // bs
            idx = range(len(X))
            for i in range(n_batches):
                sl = slice(i * bs, (i + 1) * bs)
                Xb = X[idx[sl]]
                yb = y[idx[sl]]
                yield self.transform(Xb, yb)

    @property
    def n_samples(self):
        X = self.X
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state
