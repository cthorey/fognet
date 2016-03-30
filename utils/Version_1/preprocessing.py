from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import pandas as pd

# Possible processing object


class BaseTransformerMixin(TransformerMixin):

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **params):
        for key, val in params.iteritems():
            setattr(self, key, val)


class MissingValueInputer(BaseTransformerMixin):

    def __init__(self, method='time'):
        self.method = method

    def transform(self, X):
        return X.interpolate(method=self.method)

    def fit(self, X, y=None):
        return self


class FillRemainingNaN(BaseTransformerMixin):

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


class MyPipeline(Pipeline):

    def df_transform(self, df):
        df_tmp = pd.DataFrame(self.transform(df),
                              columns=df.columns,
                              index=df.index)
        return df_tmp

# Build the pipe


def build_pipeline(pipe_list, pipe_kwargs):
    try:
        steps = [(name, globals()[name]()) for name in pipe_list]
    except:
        raise ValueError('You pipe_list is fucked up')
    pipe = MyPipeline(steps)
    pipe.set_params(**pipe_kwargs)
    return pipe
