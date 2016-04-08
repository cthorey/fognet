import os
import statsmodels.api as sm
from scipy.stats import normaltest
# import matplotlib.pylab as plt

from train_utils import BaseModel
from preprocessing import *
from data_utils import *
from helper import *

import pprint
from sklearn.metrics import mean_squared_error


class ArimaModel(BaseModel):
    ''' class to handle ARIMA model '''

    def __init__(self, config, hp=['AR', 'D', 'MA'], mode='train'):
        super(ArimaModel, self).__init__(config=config, mode=mode, hp=hp)

    def init_data(self):
        ################################################################
        # Load the preprocessing
        print 'Loading the prepro pipeline'
        # pprint.pprint(self.pipe)
        df = add_group_column_to_data(build_dataset())

        # Process the feature space
        self.pipeline = build_entire_pipeline(
            self.pipe['pipe_list'],
            self.pipe['pipe_kwargs'],
            df_indexer=df,
            pca_components=self.pca_components)
        self.df = self.pipeline.fit_transform(df)
        self.regressors = [f for f in self.df.columns if f.split('_')[
            0] == 'feat' and f.split('_')[-1] != 'yield']

        # transform the yield
        self.pipeline_yield = build_entire_pipeline(
            self.pipe_yield['pipe_list'],
            self.pipe_yield['pipe_kwargs'],
            df_indexer=self.df,
            pca_components=0)
        self.df = self.pipeline_yield.fit_transform(self.df)

        # Add a coolumn of zero for the prediction
        self.df['yield_pred'] = 0

    def init_model(self, mode='train'):

        self.architecture = getattr(sm.tsa, self.which_architecture)
        self.order = (self.AR, self.D, self.MA)
        self.seasonal_order = (
            self.Season_AR, self.Season_D, self.Season_MA, self.Season_Period)
        if mode == 'train':
            print 'Set up the checkpoints'
            self.init_checkpoints()

    def get_model(self, df, enforce_stationarity=True, enforce_invertibility=True):
        if self.which_architecture == 'SARIMAX':
            return self.architecture(endog=df['feat_yield'],
                                     exog=df[self.regressors],
                                     order=self.order,
                                     seasonal_order=self.seasonal_order,
                                     enforce_invertibility=enforce_invertibility,
                                     enforce_stationarity=enforce_stationarity)
        else:
            raise ValueError('%s is not implemented' %
                             (self.which_architecture))

    def replace_nan(self, x):
        if np.isnan(x):
            return 1e5
        else:
            return x

    def is_there_some_nan_fit(self, fit_results):
        return any(np.isnan(np.array((fit_results.aic,
                                      fit_results.bic,
                                      fit_results.hqic,
                                      fit_results.nobs))))

    def get_information_fit(self, df, fit_results):
        return map(self.replace_nan, (self.get_score(df),
                                      fit_results.aic,
                                      fit_results.bic,
                                      fit_results.hqic,
                                      fit_results.nobs))

    def merge_fitted_values(self, df, results):
        dffitted = pd.DataFrame(np.maximum(
            0, results.fittedvalues), columns=['yield_pred'])
        return df.join(dffitted, how='left', lsuffix='l')

    def fit_group(self, df, disp=0, maxiter=100):
        # traning
        train, test = train_test_split(df)

        try:
            train_model = self.get_model(train)
            train_results = train_model.fit(maxiter=maxiter, disp=disp)
            if self.is_there_some_nan_fit(train_results):
                raise ValueError
            else:
                if self.verbose > 1:
                    print(train_results.summary())

        except ValueError:
            train_model = self.get_model(
                train, enforce_stationarity=False, enforce_invertibility=False)
            train_results = train_model.fit(maxiter=maxiter, disp=disp)
        except:
            print 'third try'
            raise ValueError()

        train = self.merge_fitted_values(train, train_results)
        train_score = self.get_information_fit(train, train_results)

        # testing
        test_model = self.get_model(test)
        test_results = test_model.filter(train_results.params)
        test = self.merge_fitted_values(test, test_results)
        test_score = self.get_information_fit(test, test_results)

        # Update the main dataframe
        df_model = self.get_model(df)
        results = df_model.filter(train_results.params).fittedvalues
        dffitted = pd.DataFrame(results, columns=['yield_pred'])
        self.df.loc[dffitted.index, 'yield_pred'] = np.maximum(
            dffitted.yield_pred, 0)

        return train_score, test_score

    def fit(self):
        train_score, test_score = [], []
        dfgroup = self.df.groupby('group')
        try:
            for name, gp in dfgroup:
                trains, tests = self.fit_group(gp)
                train_score.append(trains)
                test_score.append(tests)
            self.make_submission(self.df)
        except:
            train_score = 1e5 * np.ones((1, 5))
            test_score = 1e5 * np.ones((1, 5))
            test_score[0] = 2

        self.get_summary(train_score, split='train')
        self.get_summary(test_score, split='test')

        self.dump_final_config_file()

    def iterative_fit(self, epsilon_threeshold):
        train_score, test_score = [], []
        dfgroup = self.df.groupby('group')
        k = 0
        epsilon = 100
        while epsilon > epsilon_threeshold:
            for name, gp in dfgroup:
                _, _ = self.fit_group(gp)
            old_value = self.df.feat_yield[np.isnan(self.df['yield'])].values
            new_value = self.df.yield_pred[np.isnan(self.df['yield'])].values
            epsilon = mean_squared_error(old_value, new_value)
            # update feat_yield
            index = self.df.feat_yield[np.isnan(self.df['yield'])].index
            self.df.loc[index, 'feat_yield'] = new_value
            print('%d ieme iterations, error %2.3f' % (k, epsilon))
            k += 1
        # Once we have reach a good value for the parameters, we fit it !
        self.fit()

    def get_summary(self, score, split='train'):
        print '%s summary:' % (split)
        score = np.mean(np.array(score), axis=0)
        score_key = ['rmse', 'aic', 'bic', 'hqic']
        for i, key in enumerate(score_key):
            setattr(self, split + '_' + key, score[i])
            print('    %s : %1.3f' % (key, score[i]))

    def dump_final_config_file(self):
        unwanted = ['df', 'architecture', 'pipeline', 'pipeline_yield']
        config = {k: v for k, v in props(
            self).iteritems() if k not in unwanted}
        dump_conf_file(config, os.path.expanduser(self.folder))

    def get_score(self, df, against='yield_pred'):

        df = df[['yield', against]].dropna()
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))
