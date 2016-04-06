import os
from helper import get_current_datetime
from data_utils import *


class BaseModel(object):
    ''' class to handle ARIMA model '''

    def __init__(self, config, mode='train', hp=['']):
        assert mode in ['train', 'inspection']
        self.conf = config
        for key, val in config.iteritems():
            setattr(self, key, val)
        self.mode = mode
        self.hp = {f: config[f] for f in hp}
        print(self.hp)
        self.init_data()
        self.init_model(mode=mode)

    def init_checkpoints(self):
        # Model checkpoints
        name = '_'.join([f + '_' + str(g) for f, g in self.hp.iteritems()])
        self.model_name = name
        self.folder = os.path.join(self.root, name)
        output_exists = os.path.isdir(os.path.expanduser(self.folder))
        if output_exists and not self.overwrite:
            print 'Model output exists. Use --overwrite'
            sys.exit(1)
        elif not output_exists:
            os.mkdir(os.path.expanduser(self.folder))

        self.model_fname = os.path.join(self.folder, 'model.pkl')

    def init_data(self):
        pass

    def init_model(self, mode='train'):
        pass

    def train(self):
        pass

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
