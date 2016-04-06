import statsmodels.api as sm
from scipy.stats import normaltest
import matplotlib.pylab as plt

from preprocessing import *
from data_utils import *
import pipe_def
from helper import get_current_datetime

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
rforecast = importr('forecast')

import pprint
from tqdm import *
from sklearn.metrics import mean_squared_error


class ArimaModel(object):
    ''' class to handle ARIMA model '''

    def __init__(self, pipe='pipe0'):
        self.folder = '/Users/thorey/Documents/project/competition/fognet/models/bbking/arima'
        self.pipe = getattr(pipe_def, pipe)
        self.init_data()

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
        df['yield'] = map(self.replace_yield, df['yield'])
        self.pipeline = build_entire_pipeline(
            self.pipe['pipe_list'], self.pipe['pipe_kwargs'], df)

        self.df = self.pipeline.fit_transform(df)
        self.regressors = [f for f in self.df.columns if f.split('_')[
            0] == 'feat']

    def fit_best_arima(self, x, xreg):
        robj.globalenv['xregressors'] = xreg
        robj.globalenv['x'] = x
        robj.r('fit <- auto.arima(x,xreg=xregressors,d=1)')
        robj.r('print(summary(fit))')
        arma = robj.r('fit$arma')
        arma_names = ['AR', 'MA', 'Seasonal_AR',
                      'Seasonal_AM', 'Period', 'S', 'Seasonal_S']
        best_arima = dict(zip(arma_names, arma))
        order = map(int, (best_arima['AR'],
                          best_arima['S'],
                          best_arima['MA']))
        seasonal_order = map(int, (best_arima['Seasonal_AR'],
                                   best_arima['Seasonal_S'],
                                   best_arima['Seasonal_AM'],
                                   best_arima['Period']))
        return order, seasonal_order

    def partial_fit(self, df, order_params='auto'):
        train, test = train_test_split(df)

        if order_params == 'auto':
            order, seasonal_order = self.fit_best_arima(
                train['yield'], train[self.regressors])
        else:
            order = order_params[0]
            seasonal_order = order_params[1]

        # traning
        train_model = sm.tsa.SARIMAX(endog=train['yield'],
                                     exog=train[self.regressors],
                                     order=order,
                                     seasonal_order=seasonal_order)
        train_results = train_model.fit()
        print(train_results.summary())
        train['yield_pred'] = train_results.get_prediction().predicted_mean
        train_score = self.get_score(train)

        # testing
        test_model = sm.tsa.SARIMAX(endog=test['yield'],
                                    exog=test[self.regressors],
                                    order=order,
                                    seasonal_order=seasonal_order)
        test_results = test_model.filter(train_results.params)
        test['yield_pred'] = test_results.get_prediction().predicted_mean
        test_score = self.get_score(test)

        # Update the main dataframe
        model = sm.tsa.SARIMAX(endog=df['yield'],
                               exog=df[self.regressors],
                               order=order,
                               seasonal_order=seasonal_order)
        results = model.filter(train_results.params)
        prediction = results.get_prediction().predicted_mean
        self.df.loc[df.index, 'yield_pred'] = prediction

        return train_score, test_score

    def fit(self):
        train_score, test_score = [], []
        self.df['yield_pred'] = 0
        dfg = self.df.groupby('group')
        for name, gp in tqdm(dfg, total=dfg.ngroups):
            trains, tests = self.partial_fit(gp,
                                             order_params=(
                                                 (0, 1, 2), (0, 0, 0, 0)))
            train_score.append(trains)
            test_score.append(tests)

        print('RMSE')
        print('RMSE - Train set : %1.3f; Test set : %1.3f' %
              (np.array(train_score).mean(), np.array(test_score).mean()))

        self.make_submission(self.df)

    def get_score(self, df):

        df = df[['yield', 'yield_pred']].dropna()
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))

    def make_submission(self, df):
        ''' Given a dataframe, make the prediction '''

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
