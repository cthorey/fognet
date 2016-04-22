from multiprocessing import cpu_count
from joblib import Parallel, delayed
from arima_utils import *
from data_utils import *
import os
from helper import *
import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error

from tqdm import *
import time


def get_best_ensemble(dfmodels, nb_models_init=25):
    print '... Loading the ensemble ...'
    ensemble = EnsembleArima(dfmodels, nb_models_init)
    print '... Looking at the resulting score ...'
    score = []
    for i in range(len(ensemble.ensemble)):
        try:
            nb_models = len(ensemble.ensemble)
            ensemble.get_score()
            score.append([nb_models, ensemble.rmse])
            ensemble.ensemble.pop()
        except:
            pass
    idx = np.argmin(np.array(score)[:, 1])
    best_nb_models = np.array(score)[idx, 0]
    print '... loading the best ensemble: n =%d ...' % (best_nb_models)
    return EnsembleArima(dfmodels, int(best_nb_models))


def get_model(dfmodels, i, verbose=0):
    ''' Need to be defined here to allow Parallel to work'''
    base_model = str(dfmodels.root.iloc[0])
    config = parse_conf_file(os.path.join(
        base_model, dfmodels.iloc[i].model, 'conf_model.json'))
    model = ArimaModel(config, mode='inspection', verbose=verbose)
    model.predict()
    return model


class EnsembleArima(object):

    def __init__(self, dfmodels, nb_models):
        self.folder = os.path.expanduser(str(dfmodels.root.iloc[0]))
        self.init_folder()
        self.dfmodels = dfmodels
        self.nb_models = nb_models
        self.get_ensemble()
        self.get_score()

    def init_folder(self):
        self.folder = os.path.join(self.folder, 'ensemble')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def get_ensemble(self, n_jobs=2):

        start_time = time.time()
        # self.ensemble = Parallel(n_jobs)(
        #     delayed(get_model)(self.dfmodels, i) for i in tqdm(range(self.nb_models), ncols=self.nb_models))
        #         self.ensemble = []
        self.ensemble = []
        for i in tqdm(range(self.nb_models)):
            self.ensemble.append(get_model(self.dfmodels, i))
        print("--- %s seconds ---" % (time.time() - start_time))

    def predict_yields(self):
        ''' Return a dataframe with the predicted yield of
        ALL the models '''
        for i, model in tqdm(enumerate(self.ensemble)):
            if i == 0:
                df = model.df[['yield', 'yield_pred']]
            else:
                df = df.join(model.df['yield_pred'],
                             how='left', rsuffix='_%d' % (i))
        return df

    def predict_yield(self):
        ''' Return a dataframe with the average predicted yield of
        all the model
        '''
        df_tmp = self.predict_yields()
        y = df_tmp['yield']
        ypred = np.array(
            df_tmp[[f for f in df_tmp.columns if len(f.split('_')) > 2 and f.split('_')[0] == 'yield']])
        ypred = np.mean(ypred, axis=1)
        df = pd.DataFrame(np.array([y, ypred]).T, columns=[
            'yield', 'yield_pred'], index=df_tmp.index)
        return df

    def get_score(self):

        df = self.predict_yield()
        self.rmse = self.get_rmse(df)
        print('RMSE = %1.3f' % (self.rmse))

    def get_rmse(self, df, against='yield_pred'):

        df = df[['yield', against]].dropna()
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))

    def make_submission(self):
        ''' Given a dataframe, make the prediction '''

        df = self.predict_yield()
        output_fname = os.path.join(
            self.folder, 'submissions_%s.csv' % get_current_datetime())
        print 'Will write output to %s' % output_fname

        ################################################################
        # Merge and produce  the submission file
        submission_df = load_raw_data()['submission_format']
        final_pred_format = submission_df.join(df, how='left', rsuffix='r')
        submission_df['yield'] = final_pred_format['yield_pred']

        ################################################################
        # Remove value below zero !
        submission_df[submission_df['yield'] < 0.0] = 0

        ################################################################
        # Store to a txt file
        submission_df.to_csv(os.path.expanduser(output_fname))


class EnsembleNet(object):

    def __init__(self, dfmodels, nb_models):
        self.folder = os.path.expanduser(str(dfmodels.root.iloc[0]))
        self.init_folder()
        self.dfmodels = dfmodels
        self.nb_models = nb_models
        self.get_ensemble()

    def init_folder(self):
        self.folder = os.path.join(self.folder, 'ensemble')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def get_ensemble(self, n_jobs=2):

        start_time = time.time()
        # self.ensemble = Parallel(n_jobs)(
        # delayed(get_model)(self.dfmodels, i) for i in
        # tqdm(range(self.nb_models), ncols=self.nb_models))
        self.ensemble = []
        for i in tqdm(range(self.nb_models)):
            self.ensemble.append(get_model(self.dfmodels, i))

        print("--- %s seconds ---" % (time.time() - start_time))

    def predict_yields(self, split='train'):
        ''' Return a dataframe with the predicted yield of
        ALL the models '''
        for i, model in enumerate(self.ensemble):
            if i == 0:
                df_tmp = model.predict_yield(split)
                df = df_tmp[['yield', 'yield_pred']]
            else:
                df_tmp = model.predict_yield(split)
                df['yield_pred%s' % (i)] = df_tmp['yield_pred']
        return df

    def predict_yield(self, split='train'):
        ''' Return a dataframe with the average predicted yield of
        all the model
         '''
        df_tmp = self.predict_yields(split)
        y = df_tmp['yield']
        ypred = np.array(
            df_tmp[[f for f in df_tmp.columns if len(f.split('_')) == 2]])
        ypred = np.mean(ypred, axis=1)
        df = pd.DataFrame(np.array([y, ypred]).T, columns=[
            'yield', 'yield_pred'], index=df_tmp.index)
        return df

    def get_score_set(self, split):
        df = self.predict_yield(split)
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))

    def make_submission(self):
        ''' Given a dataframe, make the prediction '''

        df = self.predict_yield(split='pred')
        output_fname = os.path.join(
            self.folder, 'submissions_%s.csv' % get_current_datetime())
        print 'Will write output to %s' % output_fname

        ################################################################
        # Merge and produce  the submission file
        submission_df = load_raw_data()['submission_format']
        final_pred_format = submission_df.join(df, how='left', rsuffix='r')
        submission_df['yield'] = final_pred_format['yield_pred']

        ################################################################
        # Remove value below zero !
        submission_df[submission_df['yield'] < 0.0] = 0

        ################################################################
        # Store to a txt file
        submission_df.to_csv(os.path.expanduser(output_fname))


class EnsembleArimaGroup(object):

    def __init__(self, dfmodels, nb_models):
        self.folder = os.path.expanduser(str(dfmodels.root.iloc[0]))
        self.init_folder()
        self.dfmodels = dfmodels
        self.nb_models = nb_models
        self.get_ensemble()
        self.get_score()

    def init_folder(self):
        self.folder = os.path.join(self.folder, 'ensemble')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def get_ensemble(self, n_jobs=2):

        start_time = time.time()
        # self.ensemble = Parallel(n_jobs)(
        #     delayed(get_model)(self.dfmodels, i) for i in tqdm(range(self.nb_models), ncols=self.nb_models))
        #         self.ensemble = []
        self.ensemble = []
        for i in tqdm(range(self.nb_models)):
            self.ensemble.append(get_model(self.dfmodels, i))
        print("--- %s seconds ---" % (time.time() - start_time))

    def predict_yields(self):
        ''' Return a dataframe with the predicted yield of
        ALL the models '''
        for i, model in tqdm(enumerate(self.ensemble)):
            if i == 0:
                df = model.df[['yield', 'yield_pred']]
            else:
                df = df.join(model.df['yield_pred'],
                             how='left', rsuffix='_%d' % (i))
        return df

    def predict_yield(self):
        ''' Return a dataframe with the average predicted yield of
        all the model
        '''
        df_tmp = self.predict_yields()
        y = df_tmp['yield']
        ypred = np.array(
            df_tmp[[f for f in df_tmp.columns if len(f.split('_')) > 2 and f.split('_')[0] == 'yield']])
        ypred = np.mean(ypred, axis=1)
        df = pd.DataFrame(np.array([y, ypred]).T, columns=[
            'yield', 'yield_pred'], index=df_tmp.index)
        return df

    def get_score(self):

        df = self.predict_yield()
        train, test = train_test_split(df, verbose=0)
        self.rmse_train = self.get_rmse(train)
        self.rmse_test = self.get_rmse(test)
        print('RMSE train split = %1.3f' % (self.rmse_train))
        print('RMSE test split = %1.3f' % (self.rmse_test))

    def get_rmse(self, df, against='yield_pred'):

        df = df[['yield', against]].dropna()
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))

    def make_submission(self):
        ''' Given a dataframe, make the prediction '''

        df = self.predict_yield()
        output_fname = os.path.join(
            self.folder, 'submissions_%s.csv' % get_current_datetime())
        print 'Will write output to %s' % output_fname

        ################################################################
        # Merge and produce  the submission file
        submission_df = load_raw_data()['submission_format']
        final_pred_format = submission_df.join(df, how='left', rsuffix='r')
        submission_df['yield'] = final_pred_format['yield_pred']

        ################################################################
        # Remove value below zero !
        submission_df[submission_df['yield'] < 0.0] = 0

        ################################################################
        # Store to a txt file
        submission_df.to_csv(os.path.expanduser(output_fname))
