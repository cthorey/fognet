import numpy as np
import pandas as pd


def build_seq(idx):
    return range(idx, idx + 48)


def train_test_split_rand_yield(df, seed=91, nb_gaps=7, verbose=1):
    ''' Return a train/val/test split of the data for training
    seed_ok with nb_gaps=7 :[5, 7, 16, 29, 91, 123, 129, 130, 150, 174]
    '''

    assert len(set(df.group)) == 1
    randgen = np.random.RandomState(seed)
    idx = np.arange(len(df))
    init_idx_test = list(
        np.sort(randgen.choice(range(len(df) - 48), size=nb_gaps, replace=False)))
    idx_test = reduce(lambda x, y: x + y,
                      map(build_seq, list(init_idx_test)))

    # duplciates ?
    assert len(set([x for x in idx_test if idx_test.count(x) > 1])) == 0
    mask = np.in1d(idx, idx_test)
    idx_train = np.sort(idx[~mask])

    train = df.copy()
    train.loc[df.iloc[idx_test].index, 'feat_yield'] = np.nan

    test = df.copy()
    test.loc[df.iloc[idx_train].index, 'feat_yield'] = np.nan

    if verbose > 0:
        print('Le train is composed by %d observation' %
              (len(train)))
        print('Le test is composed %d observation' %
              (len(test)))

    return train, test


def train_test_split_rand(df, seed=103, verbose=1):
    ''' Return a train/val/test split of the data for training '''

    idx = np.arange(len(gp))
    dates = pd.date_range(gp.index.min(), gp.index.max(), freq='2H')
    idx_train = np.sort(randgen.choice(
        idx, size=int(len(idx) * split), replace=False))
    mask = np.in1d(idx, idx_train)
    idx_test = np.sort(idx[~mask])
    train = gp.iloc[idx_train]
    test = gp.iloc[idx_test]

    if verbose > 0:
        print('Le train is composed by %d observation' %
              (len(train)))
        print('Le test is composed %d observation' %
              (len(test)))
    return train, test


def train_test_split_base(df, verbose=1):
    ''' Return a train/val/test split of the data for training '''

    train = df.iloc[:int(len(gp) * 0.75)]
    test = df.iloc[int(len(gp) * 0.75):]

    if verbose > 0:
        print('Le train is composed by %d observation' %
              (len(train)))
        print('Le test is composed %d observation' %
              (len(test)))
    return train, test
