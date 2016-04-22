import os
import statsmodels.api as sm
from scipy.stats import normaltest
# import matplotlib.pylab as plt

from train_utils import BaseModel
from preprocessing import *
from data_utils import *
from helper import *
from hook import SaveArimaParameters
from split_method import *
from tqdm import *
import pickle
import random
import pprint
from sklearn.metrics import mean_squared_error


class Pipe(object):

    def __call__(self, pipe_list, pipe_kwargs):
        return {'pipe_list': pipe_list,
                'pipe_kwargs': pipe_kwargs}


class ArimaModel(BaseModel):
    ''' class to handle ARIMA model '''

    def __init__(self, config, hp=['AR', 'D', 'MA'], mode='train', verbose=1, **kwargs):
        if not all(map(lambda x: x in config.keys(), kwargs.keys())):
            raise ValueError()
        else:
            config.update(kwargs)
            if len(kwargs.keys()) != 0:
                hp = kwargs.keys()
        super(ArimaModel, self).__init__(
            config, mode=mode, hp=hp, verbose=verbose)
        self.init_pipe()
        self.init_data()
        self.init_model(mode=mode)
        self.init_parameter_saver()

    def init_pipe(self):
        mpipe = Pipe()
        # Pipe_list_0
        # Lags normal
        pipe_list_0 = ['FeatureSelector',
                       str(self.inputer),
                       'CreateLagArrays',
                       'FillRemainingNaN',
                       'StandardScaler']
        pipe_kwargs_0 = {'FeatureSelector__features': self.features_base,
                         'CreateLagArrays__lags': self.num_lags_regressors,
                         'CreateLagArrays__inter_lags': self.seasonal_inter_lags}
        pipe_list = {'pipe_0': pipe_list_0}
        pipe_kwargs = {'pipe_0': pipe_kwargs_0}

        if self.num_features_extra > 0:
            # Pipe_list_1
            pipe_list_1 = pipe_list_0
            feature_extra = [
                f for f in self.numerical_feature if f not in self.features_base]
            self.features_extra = random.sample(
                feature_extra, self.num_features_extra)
            pipe_kwargs_1 = {'FeatureSelector__features': self.features_extra,
                             'CreateLagArrays__lags': self.num_lags_regressors,
                             'CreateLagArrays__inter_lags': self.seasonal_inter_lags}
            pipe_list.update({'pipe_1': pipe_list_1})
            pipe_kwargs.update({'pipe_1': pipe_kwargs_1})

        self.pipe = mpipe(pipe_list, pipe_kwargs)

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
        if self.mode=='train':
            setattr(self, 'save',
                    SaveArimaParameters(getattr(self, 'model')))

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
            path = os.path.join(self.folder, 'model.pkl')
            setattr(self, 'model', path)

        if mode == 'inspection':
            path = os.path.expanduser(os.path.join(self.root,self.model_name,'model.pkl'))
            if self.verbose > 0:
                print 'Loading model params for %s' % (path)
            with open(path) as f:
                setattr(self, 'model', pickle.load(f))

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

    def merge_fitted_values(self, df, fittedvalues):
        dffitted = pd.DataFrame(np.maximum(
            0, fittedvalues), columns=['yield_pred'])
        return df.join(dffitted, how='left', lsuffix='l')

    def fit(self, df, disp=0, maxiter=100):
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

        return df_results

    def save_model_params(self, df_results):
        getattr(self, 'save')(df_results)

    def load_model_params(self, df_results):
        setattr(self, 'model', df_results)

    def get_model(self, df, fitted_model):
        model = self.get_model_architecture(df)
        return model.filter(fitted_model.params)

    def get_scores(self, name_group, df):
        model = self.get_model(name_group, df)
        df = self.merge_fitted_values(df, model.fittedvalues)
        return self.get_information_fit(df, model)

    def get_dfpred(self, name_group, df):
        model = self.get_model(name_group, df)
        return self.merge_fitted_values(df, model.fittedvalues)

    def update_main_df(self, df, fitted_model):
        df = self.merge_fitted_values(df, fitted_model.fittedvalues)
        self.df.loc[df.index, 'yield_pred'] = df.yield_pred.values

    def predict(self, nb_folds=10, size_gap=96, seed=91):

        gp = self.df.copy()
        model = self.get_model(gp,self.model)
        self.update_main_df(gp,model)

    def get_information_fit(self, df, fit_results):
        return map(replace_nan, (self.get_rmse(df),
                                 fit_results.aic,
                                 fit_results.bic,
                                 fit_results.hqic,
                                 fit_results.nobs))

    def train_CV(self, nb_folds=10, size_gap=96, seed=91):
        if self.mode != 'train':
            raise ValueError('run in training mode : mode=train')
        randgen = np.random.RandomState(seed)
        gp = self.df.copy()
        n = len(gp)
        # CV
        try:
            idx = np.array(range(96 * 2, n))
            mask = np.array(gp.iloc[96 * 2:n]['yield'].apply(np.isnan))
            possible_values = idx[~mask]  # idx where yield is not nan
            idx = list(np.sort(randgen.choice(
                possible_values, size=nb_folds, replace=False)))
            train_score = []
            test_score = []
            for i in tqdm(range(nb_folds)):
                train, test = gp.iloc[:idx[i]], gp.iloc[
                    idx[i]:idx[i] + size_gap]
                # fit the model
                train_model = self.fit(train)
                train.loc[train.index, 'yield_pred'] = train_model.fittedvalues
                # get the score
                train_score.append(
                    self.get_information_fit(train, train_model))
                # get the score
                test_model = self.get_model(test, train_model)
                test.loc[test.index, 'yield_pred'] = test_model.fittedvalues
                test_score.append(
                    self.get_information_fit(test, test_model))
            final_model = self.fit(gp)
            self.save_model_params(final_model)
            self.update_main_df(gp, final_model)
            self.get_summary(train_score, split='train', CV=True)
            self.get_summary(test_score, split='test', CV=True)
            self.make_submission(self.df)
        except:
            train_score = 1e5 * np.ones((1, 5))
            test_score = 1e5 * np.ones((1, 5))
            test_score[0] = 3
            self.get_summary(train_score, split='train')
            self.get_summary(test_score, split='test')
        self.dump_final_config_file()

    def get_summary(self, score, split='train', CV=True):
        if self.verbose > 0:
            print '%s summary:' % (split)
        score_mean = np.mean(np.array(score), axis=0)
        score_std = np.std(np.array(score), axis=0)
        score_key = ['rmse', 'aic', 'bic', 'hqic']
        for i, key in enumerate(score_key):
            if CV:
                setattr(self, 'CV_' + split + '_' + key, score_mean[i])
                setattr(self, 'CV_std' + split + '_' + key, score_std[i])
            if self.verbose > 0:
                print('    %s : %1.3f, %1.3f' %
                      (key, score_mean[i], score_std[i]))

    def dump_final_config_file(self):
        unwanted = ['df', 'architecture',
                    'pipeline', 'pipeline_yield', 'dfgroup']
        unwanted += ['save', 'model']
        config = {k: v for k, v in props(
            self).iteritems() if k not in unwanted}
        dump_conf_file(config, os.path.expanduser(self.folder))

    def get_rmse(self, df, against='yield_pred'):

        dff = df[['yield', against]].dropna()
        try:
            return np.sqrt(mean_squared_error(dff['yield'], dff['yield_pred']))
        except:
            print df['yield_pred']
            return 2


class ArimaModelGroup(BaseModel):
    ''' class to handle ARIMA model '''

    def __init__(self, config, hp=['AR', 'D', 'MA'], mode='train', verbose=1, **kwargs):
        if not all(map(lambda x: x in config.keys(), kwargs.keys())):
            raise ValueError()
        else:
            config.update(kwargs)
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

        return df_results

    def save_model_params(self, name, df_results):
        getattr(self, 'save_' + name)(df_results)

    def load_model_params(self, name, df_results):
        setattr(self, 'model_' + name, df_results)

    def iterative_fit(self, name, df, epsilon_threeshold):
        k = 0
        epsilon = 100
        while epsilon > epsilon_threeshold:

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

    def get_model(self, df, fitted_model):
        model = self.get_model_architecture(df)
        return model.filter(fitted_model.params)

    def get_scores(self, name_group, df):
        model = self.get_model(name_group, df)
        df = self.merge_fitted_values(df, model.fittedvalues)
        return self.get_information_fit(df, model)

    def get_dfpred(self, name_group, df):
        model = self.get_model(name_group, df)
        return self.merge_fitted_values(df, model.fittedvalues)

    def update_main_df(self, df, fitted_model):
        df = self.merge_fitted_values(df, fitted_model.fittedvalues)
        self.df.loc[df.index, 'yield_pred'] = df.yield_pred.values

    def predict(self):
        train_score, test_score = [], []
        for name, gp in self.dfgroup:
            train, test = train_test_split_rand_yield(gp, verbose=self.verbose)
            train_score.append(self.get_scores(name, train))
            test_score.append(self.get_scores(name, test))
            self.update_main_df(name, gp)

        self.get_summary(train_score, split='train')
        self.get_summary(test_score, split='test')

    def get_information_fit(self, df, fit_results):
        return map(replace_nan, (self.get_rmse(df),
                                 fit_results.aic,
                                 fit_results.bic,
                                 fit_results.hqic,
                                 fit_results.nobs))

    def train_CV(self, nb_folds=5, size_gap=96, seed=91):
        if self.mode != 'train':
            raise ValueError('run in training mode : mode=train')
        randgen = np.random.RandomState(seed)
        train_score = []
        test_score = []
        for name, gp in self.dfgroup:
            n = len(gp)
            print name, n
            # idx = range(init_len, len(gp), gap) + [len(gp)]  # Grid for the
            # CV
            idx = np.array(range(96 * 2, n))
            mask = np.array(gp.iloc[96 * 2:n]['yield'].apply(np.isnan))
            possible_values = idx[~mask]  # idx where yield is not nan
            # possible_values = range(len(gp) - 2 * size_gap)
            idx = list(np.sort(randgen.choice(
                possible_values, size=nb_fold, replace=False)))
            print idx
            train_score_tmp = []
            test_score_tmp = []
            for i in tqdm(range(nb_fold)):
                train, test = gp.iloc[:idx[i]], gp.iloc[
                    idx[i]:idx[i] + size_gap]
                # fit the model
                train_model = self.fit(name, train)
                train.loc[train.index, 'yield_pred'] = train_model.fittedvalues
                # get the score
                train_score_tmp.append(
                    self.get_information_fit(train, train_model))
                # get the score
                test_model = self.get_model(test, train_model)
                test.loc[test.index, 'yield_pred'] = test_model.fittedvalues
                test_score_tmp.append(
                    self.get_information_fit(test, test_model))
            final_model = self.fit(name, gp)
            self.save_model_params(name, final_model)
            self.update_main_df(gp, final_model)
            print np.array(test_score_tmp).std(axis=0)
            print np.array(test_score_tmp).mean(axis=0)
            train_score.append(np.array(train_score_tmp).mean(axis=0))
            test_score.append(np.array(test_score_tmp).mean(axis=0))
        self.get_summary(train_score, split='train', CV=True)
        self.get_summary(test_score, split='test', CV=True)

        self.make_submission(self.df)

        self.dump_final_config_file()

    def train(self):
        if self.mode != 'train':
            raise ValueError('run in training mode : mode=train')

        try:
            for name, gp in self.dfgroup:
                train, _ = train_test_split_rand_yield(gp)
                # fit the model
                train_model = self.fit(name, train)
                self.get_rmse(train.feat_yield, train_model.fittedvalues)
                # get the score
                test_model = self.get_model(test, train_model)
                self.get_rmse(test.feat_yield, test_model.fittedvalues)

            self.predict()
            self.make_submission(self.df)
        except:
            train_score = 1e5 * np.ones((1, 5))
            test_score = 1e5 * np.ones((1, 5))
            test_score[0] = 2
            self.get_summary(train_score, split='train')
            self.get_summary(test_score, split='test')
        self.dump_final_config_file()

    def get_summary(self, score, split='train', CV=False):
        if self.verbose > 0:
            print '%s summary:' % (split)
        score = np.mean(np.array(score), axis=0)
        score_key = ['rmse', 'aic', 'bic', 'hqic']
        for i, key in enumerate(score_key):
            if CV:
                setattr(self, 'CV_' + split + '_' + key, score[i])
            else:
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

        dff = df[['yield', against]].dropna()
        try:
            return np.sqrt(mean_squared_error(dff['yield'], dff['yield_pred']))
        except:
            print df['yield_pred']
            return 2
