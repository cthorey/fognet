import os
import statsmodels.api as sm
from scipy.stats import normaltest
# import matplotlib.pylab as plt

from train_utils import BaseModel
from preprocessing import *
from data_utils import *
from helper import *
from hook import SaveArimaParameters

import pickle
import pprint
from sklearn.metrics import mean_squared_error


class ArimaModel(BaseModel):
    ''' class to handle ARIMA model '''

    def __init__(self, config, hp=['AR', 'D', 'MA'], mode='train', verbose=1):
        super(ArimaModel, self).__init__(
            config, mode=mode, hp=hp, verbose=verbose)
        self.init_data()
        self.init_model(mode=mode)
        self.init_parameter_saver()

    def init_data(self):
        ################################################################
        # Load the preprocessing
        if self.verbose > 0:
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
        self.dfgroup = self.df.groupby('group')
        self.ngroups = self.dfgroup.ngroups

    def init_parameter_saver(self):

        for name in self.dfgroup.groups.keys():
            setattr(self, 'save_' + name,
                    SaveArimaParameters(getattr(self, 'model_f' + name)))

    def init_model(self, mode='train'):

        self.architecture = getattr(sm.tsa, self.which_architecture)
        self.order = (self.AR, self.D, self.MA)
        self.seasonal_order = (
            self.Season_AR, self.Season_D, self.Season_MA, self.Season_Period)
        if self.verbose > 0:
            print 'Order : '
            print self.order
            print 'Season order : '
            print self.seasonal_order
        if mode == 'train':
            print 'Set up the checkpoints'
            self.init_checkpoints()
            for group in self.dfgroup.groups.keys():
                path = os.path.join(self.folder, 'model_' + group + '.pkl')
                setattr(self, 'model_f' + group, path)

        if mode == 'inspection':
            for name in self.dfgroup.groups.keys():
                path = os.path.expanduser(getattr(self, 'model_f' + name))
                if self.verbose > 0:
                    print 'Loading model params for %s from %s' % (name, path)
                with open(path) as f:
                    setattr(self, 'model_' + name, pickle.load(f))

    def get_model_architecture(self, df, enforce_stationarity=True, enforce_invertibility=True):
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

    def is_there_some_nan_fit(self, fit_results):
        return any(np.isnan(np.array((fit_results.aic,
                                      fit_results.bic,
                                      fit_results.hqic,
                                      fit_results.nobs))))

    def get_information_fit(self, df, fit_results):
        return map(replace_nan, (self.get_rmse(df),
                                 fit_results.aic,
                                 fit_results.bic,
                                 fit_results.hqic,
                                 fit_results.nobs))

    def merge_fitted_values(self, df, fittedvalues):
        dffitted = pd.DataFrame(np.maximum(
            0, fittedvalues), columns=['yield_pred'])
        return df.join(dffitted, how='left', lsuffix='l')

    def fit(self, name, df, disp=0, maxiter=100):
        try:
            df_model = self.get_model_architecture(df)
            df_results = df_model.fit(maxiter=maxiter, disp=0, iprint=0)
            if self.is_there_some_nan_fit(df_results):
                raise ValueError
            else:
                if self.verbose > 1:
                    print(df_results.summary())

        except ValueError:
            df_model = self.get_model_architecture(
                df, enforce_stationarity=False, enforce_invertibility=False)
            df_results = df_model.fit(maxiter=maxiter, disp=0, iprint=0)
        except:
            print 'third try'
            raise ValueError()
        setattr(self, 'model_' + name, df_results)
        getattr(self, 'save_' + name)(df_results)

    def iterative_fit(self, epsilon_threeshold):
        train_score, test_score = [], []
        dfgroup = self.df.groupby('group')
        k = 0
        epsilon = 100
        while epsilon > epsilon_threeshold:
            for name, gp in dfgroup:
                _, _ = self.fit_group(name, gp)
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

    def get_model(self, name_group, df):
        fitted_model = getattr(self, 'model_' + name_group)
        model = self.get_model_architecture(df)
        return model.filter(fitted_model.params)

    def get_scores(self, name_group, df):
        model = self.get_model(name_group, df)
        df = self.merge_fitted_values(df, model.fittedvalues)
        return self.get_information_fit(df, model)

    def get_dfpred(self, name_group, df):
        model = self.get_model(name_group, df)
        return self.merge_fitted_values(df, model.fittedvalues)

    def update_main_df(self, name_group, df):
        model = self.get_model(name_group, df)
        df = self.merge_fitted_values(df, model.fittedvalues)
        self.df.loc[df.index, 'yield_pred'] = df.yield_pred.values

    def predict(self):
        train_score, test_score = [], []
        for name, gp in self.dfgroup:
            train, test = train_test_split(gp, verbose=self.verbose)
            train_score.append(self.get_scores(name, train))
            test_score.append(self.get_scores(name, test))
            self.update_main_df(name, gp)

        self.get_summary(train_score, split='train')
        self.get_summary(test_score, split='test')

    def train(self):
        if self.mode != 'train':
            raise ValueError('run in training mode : mode=train')

        try:
            for name, gp in self.dfgroup:
                train, _ = train_test_split(gp)
                # fit the model
                self.fit(name, train)
                # get the score

            self.predict()
            self.make_submission(self.df)
        except:
            train_score = 1e5 * np.ones((1, 5))
            test_score = 1e5 * np.ones((1, 5))
            test_score[0] = 2
            self.get_summary(train_score, split='train')
            self.get_summary(test_score, split='test')
        self.dump_final_config_file()

    def get_summary(self, score, split='train'):
        if self.verbose > 0:
            print '%s summary:' % (split)
        score = np.mean(np.array(score), axis=0)
        score_key = ['rmse', 'aic', 'bic', 'hqic']
        for i, key in enumerate(score_key):
            setattr(self, split + '_' + key, score[i])
            if self.verbose > 0:
                print('    %s : %1.3f' % (key, score[i]))

    def dump_final_config_file(self):
        unwanted = ['df', 'architecture',
                    'pipeline', 'pipeline_yield', 'dfgroup']
        unwanted += ['save_' + group for group in self.dfgroup.groups.keys()]
        unwanted += ['model_' + group for group in self.dfgroup.groups.keys()]
        config = {k: v for k, v in props(
            self).iteritems() if k not in unwanted}
        dump_conf_file(config, os.path.expanduser(self.folder))

    def get_rmse(self, df, against='yield_pred'):

        df = df[['yield', against]].dropna()
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))
