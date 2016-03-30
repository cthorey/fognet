import pandas as pd
import numpy as np
from iterator import BaseBatchIterator
import os
from os.path import expanduser
home = expanduser("~")
fognet = os.path.join(home, 'Documents', 'project', 'competition', 'fognet')


def load_raw_data():
    ''' load the raw data'''

    name = os.path.join(
        fognet, 'data', '8272b5ce-19e4-4dbd-80f2-d47e14786fa2.csv')
    microclimat_train_5m = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(
        fognet, 'data', 'ba94c953-7381-4d3c-9cd9-a8c9642ecff5.csv')
    microclimat_test_5m = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(
        fognet, 'data', 'eaa4fe4a-b85f-4088-85ee-42cabad25c81.csv')
    microclimat_train = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(
        fognet, 'data', 'fb38df29-e4b7-4331-862c-869fac984cfa.csv')
    microclimat_test = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(
        fognet, 'data', '41d4a6af-93df-48ab-b235-fd69c8e5dab9.csv')
    macro_guelmim = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(
        fognet, 'data', 'b57b4f6f-8aae-4630-9a14-2d24902ddf30.csv')
    macro_sidi = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(
        fognet, 'data', 'e384729e-3b9e-4f53-af1b-8f5449c69cb7.csv')
    macro_aga = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(
        fognet, 'data', 'a0f785bc-e8c7-4253-8e8a-8a1cd0441f73.csv')
    labels = pd.read_csv(name, index_col=0, parse_dates=[0])

    name = os.path.join(fognet, 'data', 'submission_format.csv')
    submission_format = pd.read_csv(name, index_col=0, parse_dates=[0])

    return {'microclimat_train': microclimat_train,
            'microclimat_test': microclimat_test,
            'microclimat_train_5m': microclimat_train_5m,
            'microclimat_test_5m': microclimat_test_5m,
            'macro_guelmim': macro_guelmim,
            'macro_sidi': macro_sidi,
            'macro_aga': macro_aga,
            'labels': labels,
            'submission_format': submission_format}


def add_group_column_to_data(df):
    ''' Get a dataframe, and return the same DF with an extra
    columns group which reference different group.

    The split is realized on time where the lag jump from 2H to more !

    More clearly, each group contains a sequence of obs separated by only
    2 hours'''

    df = df.sort_index()
    # Get the time difference between obs
    timedelta = [df.index[i + 1] - df.index[i] for i in range(len(df) - 1)]
    # Get the idx where the hour difference is larger than 16 days !
    cut_day_mask = (np.array(map(lambda x: x.components.days, timedelta)) > 16)
    cut_hour_mask = (
        np.array(map(lambda x: x.components.hours, timedelta)) != 2)
    cut = list(np.where((cut_day_mask) | (cut_hour_mask))[0])
    dict_cut = {cut[k]: 'group' + str(k) for k in range(len(cut))}

    def idx_group(idx):
        ''' from the idx in the df, return the corresponding group '''
        try:
            mask = np.where((np.array(cut) - idx) >= 0)[0]
            group_key = np.array(cut)[mask].min()
            return dict_cut[group_key]
        except:
            return 'group' + str(len(cut))

    df['group'] = map(idx_group, range(len(df)))
    # Then remove group to small to be use
    dfgp = df.groupby('group')
    good_group = []
    for gp in set(df.group):
        if len(dfgp.get_group(gp)) > 15:
            good_group.append(gp)
    return df[df.group.isin(good_group)]


def train_val_test_split(df):
    ''' Return a train/val/test split of the data for training '''

    df = add_group_column_to_data(df)
    n = df.groupby('group').ngroups

    train = []
    val = []
    test = []
    for name, gp in df.groupby('group'):
        train.append(gp.iloc[:int(len(gp) * 0.6)])
        val.append(gp.iloc[int(len(gp) * 0.6):int(len(gp) * 0.9)])
        test.append(gp.iloc[int(len(gp) * 0.9):])

    train = reduce(lambda a, b: a.append(b), train)
    val = reduce(lambda a, b: a.append(b), val)
    test = reduce(lambda a, b: a.append(b), test)
    print('Le train is composed by %d group and %d observation' %
          (train.groupby('group').ngroups, len(train)))
    print('Le val is composed by %d group and %d observation' %
          (val.groupby('group').ngroups, len(val)))
    print('Le test is composed by %d group and %d observation' %
          (test.groupby('group').ngroups, len(test)))

    return train, val, test


def build_dataset(name):
    ''' build micro data
    Be carrefull, all the submission date are not contained
    in the test set. Maybe use interpilation !
    '''

    data = load_raw_data()
    if name == 'micro':
        train = data['microclimat_train']
        train['type'] = 'training'
        labels = data['labels']
        test = data['microclimat_test']
        sub = data['submission_format']
        sub['type'] = 'prediction'
        sub = sub.drop('yield', axis=1)
        df = train.append(sub.join(test, how='left'))
        df = df.join(labels, how='left')
        df['yield'] = df['yield'].fillna(-1)
        df = df.sort_index()
    else:
        raise ValueError('The data format %s is not impemented yet' % (name))
    return df


class Data(object):

    def __init__(self, name, feats, pipeline, batch_size, seq_length, stride):
        self.data = load_raw_data()
        self.df = build_dataset(name)
        self.pipeline = pipeline
        self.feats = feats
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.stride = stride

    def benchmark(self, n_obs=12):
        ''' process data for the benchmark

        n_obs : Number of observation to incorporate before
        and after for the testing.
        '''

        train, val, test = train_val_test_split(self.df)

        # training inputer
        self.pipeline.fit(train[self.feats])

        # Transform the dataframe
        non_feats = [f for f in train.columns if f not in self.feats]
        train_tmp = self.pipeline.df_transform(
            train[self.feats]).join(train[non_feats])

        val_tmp = self.pipeline.df_transform(
            val[self.feats]).join(val[non_feats])

        test_tmp = self.pipeline.df_transform(
            test[self.feats]).join(test[non_feats])

        iter_kwargs = dict(feats=self.feats,
                           label='yield',
                           batch_size=self.batch_size,
                           size_seq=self.seq_length,
                           stride=self.stride)
        batch_ite_train = BaseBatchIterator(**iter_kwargs)(train_tmp)
        batch_ite_val = BaseBatchIterator(**iter_kwargs)(val_tmp)
        batch_ite_test = BaseBatchIterator(**iter_kwargs)(test_tmp)

        return len(self.feats), batch_ite_train, batch_ite_val, batch_ite_test


def load_data(name='micro',
              feats=['humidity', 'temp'],
              build_ite='benchmark',
              pipeline='base',
              batch_size=25,
              seq_length=25,
              stride=1):
    ''' load the data according to the desire processing
    return batch iterator for train/test split !'''
    data = Data(name=name,
                feats=feats,
                pipeline=pipeline,
                batch_size=batch_size,
                seq_length=seq_length,
                stride=stride)
    return getattr(data, build_ite)()


if __name__ == '__main__':
    pass
