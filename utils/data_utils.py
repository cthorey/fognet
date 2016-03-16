import pandas as pd
import numpy as np


def load_raw_data():
    ''' load the raw data'''

    micro_train_5m = pd.read_csv('data/8272b5ce-19e4-4dbd-80f2-d47e14786fa2.csv',
                                 index_col=0, parse_dates=[0])

    micro_test_5m = pd.read_csv('data/ba94c953-7381-4d3c-9cd9-a8c9642ecff5.csv',
                                index_col=0, parse_dates=[0])

    micro_train = pd.read_csv('data/eaa4fe4a-b85f-4088-85ee-42cabad25c81.csv',
                              index_col=0, parse_dates=[0])

    micro_test = pd.read_csv('data/fb38df29-e4b7-4331-862c-869fac984cfa.csv',
                             index_col=0, parse_dates=[0])

    macro_guelmim = pd.read_csv('data/41d4a6af-93df-48ab-b235-fd69c8e5dab9.csv',
                                index_col=0, parse_dates=[0])
    macro_sidi = pd.read_csv('data/b57b4f6f-8aae-4630-9a14-2d24902ddf30.csv',
                             index_col=0, parse_dates=[0])
    macro_aga = pd.read_csv('data/e384729e-3b9e-4f53-af1b-8f5449c69cb7.csv',
                            index_col=0, parse_dates=[0])

    labels = pd.read_csv('data/a0f785bc-e8c7-4253-8e8a-8a1cd0441f73.csv',
                         index_col=0, parse_dates=[0])

    submission_format = pd.read_csv('data/submission_format.csv',
                                    index_col=0, parse_dates=[0])

    return {'micro_train': micro_train,
            'micro_test': micro_test,
            'micro_train_5m': micro_train_5m,
            'micro_test_5m': micro_test_5m,
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

    # Get the time difference between obs
    timedelta = [df.index[i + 1] - df.index[i] for i in range(len(df) - 1)]
    # Get the idx where the time difference is larger than only 2H
    cut = list(
        np.where(np.array(map(lambda x: x.components.days, timedelta)) != 0)[0])
    # Put that in a dict with groups
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
    return df


def train_val_test_split(df):
    ''' Return a train/val/test split of the data '''

    df = add_group_column_to_data(df)
    group_train = ['group' + str(i) for i in range(20)]
    # I remove it as there is only Nan in that group
    group_train.remove('group0')
    train = df[df.group.isin(group_train)]
    train = train.join(labels[labels.index.isin(train.index)])
    print('Le train is composed by %d group and %d observation' %
          (train.groupby('group').ngroups, len(train)))

    group_val = ['group' + str(i) for i in range(20, 27)]
    val = df[df.group.isin(group_val)]
    val = val.join(labels[labels.index.isin(val.index)])
    print('Le val is composed by %d group and %d observation' %
          (val.groupby('group').ngroups, len(val)))

    group_test = ['group' + str(i) for i in range(27, 27 + 8)]
    test = df[df.group.isin(group_test)]
    test = test.join(labels[labels.index.isin(test.index)])
    print('Le test is composed by %d group and %d observation' %
          (test.groupby('group').ngroups, len(test)))


if __name__ == '__main__':
    pass
