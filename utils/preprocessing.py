from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Possible processing object


class MyImputer(Imputer):
    pass


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
