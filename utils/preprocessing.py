from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

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


class AutoArimaInputer(TransformerMixin, BaseEstimator):

    def transform(self, X):
        array = np.array(X)
        bestfit = np.apply_along_axis(self.fit_best_ARIMA, axis=0, arr=array)
        return bestfit

    def fit(self, X, y=None):
        return self

    def fit_best_ARIMA(self, x):
        robj.globalenv['x'] = x
        robj.r('y<-x')
        robj.r('fit <- auto.arima(x)')
        robj.r('kr <- KalmanRun(x, fit$model)')
        robj.r('id.na <- which(is.na(x))')
        robj.r('for (i in id.na) y[i] <- fit$model$Z %*% kr$states[i,]')
        return rpyn.ri2py(robj.r['y'])


class VARMAXStabilizer(TransformerMixin, BaseEstimator):

    def transform(self, X):
        model = sm.tsa.VARMAX(X)
        res = model.fit()
        return res.fittedvalues

    def fit(self, X, y=None):
        return self


class MissingValueInputer(TransformerMixin, BaseEstimator):

    def __init__(self, method='time'):
        self.method = method

    def transform(self, X):
        return X.interpolate(method=self.method)

    def fit(self, X, y=None):
        return self


class FillRemainingNaN(TransformerMixin, BaseEstimator):

    def __init__(self, method='bfill'):
        self.method = method

    def transform(self, X):
        Xnew = X.fillna(method=self.method)
        # make sur there is no more NaN in there !
        assert len(Xnew.dropna()) == len(Xnew)
        return Xnew

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
    try:
        assert pipe_list.keys() == pipe_kwargs.keys()
        features = FeatureUnion([(key, build_one_pipeline(
            pipe_list[key], pipe_kwargs[key])) for key in pipe_list.keys()])
    except:
        raise ValueError('You pipe_list is fucked up')

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
