import os
import statsmodels.api as sm
from scipy.stats import normaltest
import matplotlib.pylab as plt

from train_utils import BaseModel
from preprocessing import *
from data_utils import *
from helper import *
import pipe_def

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
rforecast = importr('forecast')

import pprint
from tqdm import *
from sklearn.metrics import mean_squared_error


class ArimaModel(BaseModel):
    ''' class to handle ARIMA model '''

    def __init__(self, config, hp=['AR', 'D', 'MA'], mode='train'):
        super(ArimaModel, self).__init__(config=config, mode=mode, hp=hp)

    def replace_yield(self, x):
        if x == -1:
            return np.nan
        else:
            return x

    def init_data(self):
        ################################################################
        # Load the preprocessing
        print 'Loading the prepro pipeline'
        # pprint.pprint(self.pipe)
        df = add_group_column_to_data(build_dataset())
        self.pipeline = build_entire_pipeline(
            self.pipe['pipe_list'], self.pipe['pipe_kwargs'], df)

        self.df = self.pipeline.fit_transform(df)
        self.regressors = [f for f in self.df.columns if f.split('_')[
            0] == 'feat']
        self.df['yield_pred'] = 0

    def init_model(self, mode='train'):

        self.architecture = getattr(sm.tsa, self.which_architecture)
        self.order = (self.AR, self.D, self.MA)
        self.seasonal_order = (
            self.Season_AR, self.Season_D, self.Season_MA, self.Season_Period)
        if mode == 'train':
            print 'Set up the checkpoints'
            self.init_checkpoints()

    def get_model(self, df):
        return self.architecture(endog=df['yield'],
                                 exog=df[self.regressors],
                                 order=self.order,
                                 seasonal_order=self.seasonal_order)

    def get_information_fit(self, df, fit_results):
        return (self.get_score(df),
                fit_results.aic,
                fit_results.bic,
                fit_results.hqic,
                fit_results.nobs)

    def fit_group(self, df):
        # traning
        train, test = train_test_split(df)

        train_model = self.get_model(train)
        train_results = train_model.fit()
        if self.verbose > 1:
            print(train_results.summary())

        train['yield_pred'] = train_results.fittedvalues
        train_score = self.get_information_fit(train, train_results)

        # testing
        test_model = self.get_model(test)
        test_results = test_model.filter(train_results.params)
        test['yield_pred'] = test_results.fittedvalues
        test_score = self.get_information_fit(test, test_results)

        # Update the main dataframe
        df_model = self.get_model(df)
        results = df_model.filter(train_results.params)
        prediction = results.fittedvalues
        self.df.loc[df.index, 'yield_pred'] = prediction

        return train_score, test_score

    def fit(self):
        train_score, test_score = [], []
        dfgroup = self.df.groupby('group')
        try:
            for name, gp in tqdm(dfgroup, total=dfgroup.ngroups):
                trains, tests = self.fit_group(gp)
                train_score.append(trains)
                test_score.append(tests)
            self.make_submission(self.df)
        except:
            train_score = 1e5 * np.ones((1, 5))
            test_score = 1e5 * np.ones((1, 5))
            test_score[0] = 10

        self.get_summary(train_score, split='train')
        self.get_summary(test_score, split='test')
        self.dump_final_config_file()

    def get_summary(self, score, split='train'):
        print '%s summary:' % (split)
        score = np.mean(np.array(score), axis=0)
        score_key = ['rmse', 'aic', 'bic', 'hqic']
        for i, key in enumerate(score_key):
            setattr(self, split + '_' + key, score[i])
            print('    %s : %1.3f' % (key, score[i]))

    def dump_final_config_file(self):
        unwanted = ['df', 'architecture', 'pipeline']
        config = {k: v for k, v in props(
            self).iteritems() if k not in unwanted}
        dump_conf_file(config, os.path.expanduser(self.folder))

    def get_score(self, df):

        df = df[['yield', 'yield_pred']].dropna()
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))
