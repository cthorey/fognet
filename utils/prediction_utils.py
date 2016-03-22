import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd
from time import strftime
from utils.training_utils import get_current_datetime
from utils.data_utils import load_raw_data


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
            dict(zip(df_pred[df_pred.group == gp].index, np.mean(ypred_reshape, axis=0))))
    final_pred = pd.DataFrame(
        final_pred.values(), index=final_pred.keys(), columns=['yield_pred'])
    return final_pred


def make_submission(config, final_pred):
    ''' Given a dataframe, make the prediction '''

    output_fname = os.path.join(
        config['folder'], 'submissions_%s.csv' % get_current_datetime())
    print 'Will write output to %s' % output_fname

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
