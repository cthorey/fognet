from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from data_utils import add_group_column_to_data

# TSa
import statsmodels.api as sm


# R stuff
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri as rpyn
pandas2ri.activate()
rforecast = importr('forecast')
# Possible processing object


class FeatureSelector(TransformerMixin, BaseEstimator):
    '''
    Feature selector class
    parameter:

    features : feature you want to select.
    '''

    def __init__(self, features=None):
        self.features = features

    def transform(self, X):
        return X[self.features]

    def fit(self, X, y=None):
        return self


class NumericFeatureSelector(TransformerMixin, BaseEstimator):
    '''
    Feature selector class. Among the feature proposed
    return only columns with numeric values !
    '''

    def transform(self, X):
        return X.select_dtypes(include=[np.number])

    def fit(self, X, y=None):
        return self


class RemoveZeroValues(TransformerMixin, BaseEstimator):

    def transform(self, X):
        return X.replace(0, np.nan)

    def fit(self, X, y=None):
        return self


class DiffTransformer(TransformerMixin, BaseEstimator):

    def transform(self, X):
        return X.diff(periods=1)

    def fit(self, X, y=None):
        return self


class AutoArimaInputer(TransformerMixin, BaseEstimator):

    def transform(self, X):
        array = np.array(X)
        return np.apply_along_axis(
            self.fit_best_ARIMA, axis=0, arr=array)

        # X = add_group_column_to_data(pd.DataFrame(X))
        # bestfits = []
        # for i, (name, gp) in enumerate(X.groupby('group')):
        #     gp_tmp = gp.drop('group', axis=1)
        #     array = np.array(gp_tmp)
        #     bestfit = np.apply_along_axis(
        #         self.fit_best_ARIMA, axis=0, arr=array)
        #     assert bestfit.shape == array.shape
        #     print bestfit.shape
        #     bestfits.append(bestfit)
        # return reduce(lambda x, y: np.vstack((x, y)), bestfits)

    def fit(self, X, y=None):
        return self

    def fit_best_ARIMA(self, x):
        'Kalman smoother do slightly better'
        robj.globalenv['x'] = x
        robj.r('y<-x')
        robj.r('fit <- auto.arima(x)')
        robj.r('kr <- KalmanRun(x, fit$model)')
        #robj.r('kr <- KalmanSmooth(x, fit$model)')
        robj.r('id.na <- which(is.na(x))')
        robj.r('for (i in id.na) y[i] <- fit$model$Z %*% kr$states[i,]')
        #robj.r('for (i in id.na) y[i] <- fit$model$Z %*% kr$smooth[i,]')
        return rpyn.ri2py(robj.r['y'])

    def fit_best_ARIMA2(self, x):
        robj.globalenv['x'] = x
        robj.r('y<-x')
        robj.r('fit <- auto.arima(x)')
        robj.r('arma <- fit$arma')
        arma = rpyn.ri2py(robj.r['arma'])
        AR, MA, S_AR, S_MA, P, D, S_D = map(int, arma)
        try:
            m = sm.tsa.SARIMAX(endog=x, order=(AR, D, MA),
                               seasonal_order=(S_AR, S_D, S_MA, P))
            res = m.fit(disp=0, iprint=0)
            output = res.fittedvalues
            print m.order, m.seasonal_order
        except ValueError:
            m = sm.tsa.SARIMAX(endog=x, order=(
                1, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
            res = m.fit(disp=0, iprint=0)
            output = res.fittedvalues
            print m.order, m.seasonal_order
        except ValueError:
            output = x
        return output


class VARMAXStabilizer(TransformerMixin, BaseEstimator):

    def transform(self, X):
        model = sm.tsa.VARMAX(X)
        res = model.fit()
        return res.fittedvalues

    def fit(self, X, y=None):
        return self


class InterpolateMissingValueInputer(TransformerMixin, BaseEstimator):

    def __init__(self, method='time'):
        self.method = method

    def transform(self, X):
        return X.interpolate(method=self.method)

    def fit(self, X, y=None):
        return self


class EWMAMissingValueInputer(TransformerMixin, BaseEstimator):

    def __init__(self, span=24, noise=False):
        self.span = span
        self.noise = noise

    def transform(self, X):
        return pd.DataFrame(X).fillna(self.ewma)

    def fit(self, X, y=None):
        self.ewma = pd.ewma(pd.DataFrame(X), span=self.span)
        return self


class CreateLagArrays(TransformerMixin, BaseEstimator):
    ''' Create lag arrays -
    params: 
    lags - number of lag vector to incorporate for each feature
    if lags =3, then 0,1,2 lags  !'''

    def __init__(self, lags=3, inter_lags=1, pad_values=np.nan):
        self.lags = lags
        self.pad_values = pad_values
        self.inter_lags = inter_lags

    def transform(self, X):
        X = np.array(X)
        print len(X.shape)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        nb_dim = X.shape[1]
        transform_array = []
        for i in range(0, nb_dim):
            transform_array.append(self.return_lags_for_one_vector(X[:, i]))
        return reduce(lambda x, y: np.vstack((x, y)), transform_array).T

    def return_lags_for_one_vector(self, x):
        ''' Get one dimensional vector and return 
        an array of lag vectors '''

        assert len(x.shape) == 1
        lag_vectors = []
        dim = x.shape[0]
        for n in range(0, self.lags, self.inter_lags):
            lag_vectors.append(np.pad(x, pad_width=(
                n, 0), mode='constant', constant_values=self.pad_values)[:dim])
        return reduce(lambda x, y: np.vstack((x, y)), lag_vectors)

    def fit(self, X, y=None):
        return self


class FillRemainingNaN(TransformerMixin, BaseEstimator):

    def __init__(self, method='bfill'):
        self.method = method

    def transform(self, X):
        df = pd.DataFrame(X)
        Xnew = df.fillna(method=self.method)
        # make sur there is no more NaN in there !
        assert len(Xnew.dropna()) == len(df)
        return np.array(Xnew)

    def fit(self, X, y=None):
        return self


class MyStandardScaler(StandardScaler):
    pass


class Indexer(TransformerMixin, BaseEstimator):
    '''
    Get the result of FeatureUnion and
    index it as the original dataframe +
    add the columns that have not been processed
    , i.e. yield, type, set ....
    '''

    def __init__(self, df, yield_pipe):
        self.df = df
        self.yield_pipe = yield_pipe

    def transform(self, X):
        nfeats = X.shape[1]
        if not self.yield_pipe:
            name_feats = ['feat_%d' % d for d in range(nfeats)]
            feat = pd.DataFrame(X, index=self.df.index, columns=name_feats)
        elif self.yield_pipe:
            name_feats = ['feat_yield']
            feat = pd.DataFrame(X, index=self.df.index, columns=name_feats)
        return feat.join(self.df, how='left')

    def fit(self, X, y=None):
        return self


# Build the pipe


def build_one_pipeline(pipe_list, pipe_kwargs):
    try:
        steps = [(name, globals()[name]()) for name in pipe_list]
    except:
        raise ValueError('You pipe_list is fucked up')
    pipe = Pipeline(steps)
    pipe.set_params(**pipe_kwargs)
    return pipe


def build_entire_pipeline(pipe_list, pipe_kwargs, df_indexer, pca_components=0):
    '''
    Build a pipeline base on first
    FeatureUnion to build the features
    Indexer to return a dataframe with the good
    index and the necessary columns
    '''
    # try:
    assert pipe_list.keys() == pipe_kwargs.keys()
    features = FeatureUnion([(key, build_one_pipeline(
        pipe_list[key], pipe_kwargs[key])) for key in pipe_list.keys()])
    # except:
    #     raise ValueError('You pipe_list is fucked up')

    if pipe_list.keys()[-1] == 'yield':
        pipe = Pipeline([
            ('feature', features),
            ('index', Indexer(df_indexer, yield_pipe=True))
        ])
    else:
        if pca_components == 0:
            pipe = Pipeline([
                ('feature', features),
                ('index', Indexer(df_indexer, yield_pipe=False))
            ])
        else:
            pipe = Pipeline([
                ('feature', features),
                ('PCA', PCA(n_components=pca_components)),
                ('index', Indexer(df_indexer, yield_pipe=False))
            ])

    return pipe
