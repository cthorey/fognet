import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd


def prediction(net, batch_iterator_pred):
    ''' Given a net and an iterator,
    return the unique set of yield  prediction along
    with the date 
    '''
    final_pred = {}
    df_pred = batch_iterator_pred.df
    for gp, X, p in batch_iterator_pred:
        mask = p[0].astype('int')
        ypred = net.predict(X)
        ypred_reshape = np.zeros(p[1])
        for k in range(ypred_reshape.shape[0]):
            ypred_reshape[k, mask[k, :]] = ypred[k, :]
        final_pred.update(
            dict(zip(df_pred[cdf_pred.group == gp].index, np.mean(ypred_reshape, axis=0))))
    final_pred = pd.DataFrame(
        final_pred.values(), index=final_pred.keys(), columns=['yield_pred'])
    return final_pred
