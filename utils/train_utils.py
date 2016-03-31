import lasagne
import importlib
from data_utils import *
import net_builder
from nolearn_net import NeuralNet
from helper import *
from preprocessing import *
import objectives_utils
from sklearn.metrics import mean_squared_error
from nolearn.lasagne.handlers import SaveWeights
from hook import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping
)
import pickle
import pprint


class Model(object):

    def __init__(self, config, mode='train', hp=['lr', 'reg', 'hiddens'], on_platform='thorey'):
        '''
        parameters:
        config : json file where the model config is stored
        mode : mode for the initialization, either train or inspection
        hp : set of hyperparameters that varies, help define the name of
        the folder where the model will be put
        on_platform : Name of the platform where the thing is done.
        '''

        assert mode in ['train', 'inspection']
        self.conf = config
        for key, val in config.iteritems():
            setattr(self, key, val)
        self.mode = mode
        self.hp = {f: config[f] for f in hp}
        print(self.hp)
        self.init_data()
        self.which_batch_iterator = {'val': self.batch_ite_val,
                                     'test': self.batch_ite_test,
                                     'train': self.batch_ite_train}
        self.on_epoch_finished = []
        self.init_model(mode=mode)

    def init_data(self):
        ################################################################
        # Load the preprocessing
        print 'Loading the prepro pipeline'
        pprint.pprint(self.pipe)
        self.df = build_dataset()
        pipeline = build_entire_pipeline(
            self.pipe['pipe_list'], self.pipe['pipe_kwargs'], self.df)

        ################################################################
        # Load the iterator
        # Initialize the batchiterator
        print 'Loading data iterator using : %s' % self.build_ite
        self.batch_ite_train, self.batch_ite_val, self.batch_ite_test = load_data(
            pipeline=pipeline,
            build_ite=self.build_ite,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            stride=self.stride)
        self.nb_features = self.batch_ite_train.nfeats

    def init_architecture(self):
        ################################################################
        # Build the architecture
        builder = getattr(net_builder, self.which_architecture)
        self.architecture = builder(n=self.nb_layers,
                                    D=self.nb_features,
                                    H=self.hiddens,
                                    grad_clip=self.grad_clip)

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
        self.model_history_fname = os.path.join(
            self.folder, 'model_history.pkl')
        self.model_graph_fname = os.path.join(self.folder, 'model_history.png')

        save_weights = SaveWeights(
            os.path.expanduser(self.model_fname), only_best=True, pickle=False)
        save_training_history = SaveTrainingHistory(
            os.path.expanduser(self.model_history_fname))
        plot_training_history = PlotTrainingHistory(
            os.path.expanduser(self.model_graph_fname))
        early_stopping = EarlyStopping(patience=self.patience)

        self.on_epoch_finished = [
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping
        ]

    def init_loss_function(self):
        self.loss_function = getattr(
            objectives_utils, self.obj_loss_function)

    def init_model(self, mode='train'):
        ''' Main function that build the model given everythin '''

        print 'Build the architecture: %s, %s' % (self.type_model, self.which_architecture)
        self.init_architecture()

        if mode == 'train':
            print 'Set up the checkpoints'
            self.init_checkpoints()

        print 'Initialize the loss function: %s' % (self.obj_loss_function)
        self.init_loss_function()
        ################################################################
        # Initialize solver
        print 'Initialize the network '
        self.net = NeuralNet(
            layers=self.architecture,
            regression=True,
            objective_loss_function=self.loss_function,
            objective_l2=self.reg,  # L2 regularization
            update=getattr(lasagne.updates, self.update_rule),
            update_learning_rate=self.lr,
            batch_iterator_train=self.batch_ite_train,
            batch_iterator_test=self.batch_ite_val,
            on_epoch_finished=self.on_epoch_finished,
            verbose=self.verbose,
            max_epochs=10000,
        )
        self.net.initialize()
        if mode == 'train':
            unwanted = ['architecture', 'batch_ite_test', 'df',
                        'batch_ite_val', 'batch_ite_train', 'conf', 'net',
                        'on_epoch_finished', 'which_batch_iterator', 'loss_function']
            config = {k: v for k, v in props(
                self).iteritems() if k not in unwanted}
            self.conf = config
            dump_conf_file(config, os.path.expanduser(self.folder))

        if mode == 'inspection':
            print 'Loading model params from %s' % self.model_fname
            self.net.load_params_from(os.path.expanduser(self.model_fname))
            with open(os.path.expanduser(self.model_history_fname)) as f:
                self.net.train_history_ = pickle.load(f)

    def train(self):
        ''' The function that allow training giving a specific model given
        a config file '''

        if self.mode != 'train':
            raise ValueError('run in training mode : mode=train')

        ################################################################
        # Reload the weights if we go from an older mode
        if self.continue_training:
            print 'Loading model params from %s' % self.model_fname
            self.net.load_params_from(os.path.expanduser(self.model_fname))
            with open(os.path.expanduser(self.model_history_fname)) as f:
                net.train_history_ = pickle.load(f)

        ###########################################################
        # Fitting
        self.net.fit(epochs=self.nb_epochs)

        ###############################################################
        # Final score
        print 'Loading best param'
        self.net.load_params_from(os.path.expanduser(self.model_fname))

        print 'Evaluating on val set'
        print self.get_score_set(split='val')

        ################################################################
        # Write final score in the folder as a name of txt file
        self.write_final_score()

        ################################################################
        # Predict the yield for the whole prediction set
        print 'Run the prediction'
        pred = self.make_prediction()
        self.make_submission(pred)

    def get_loss(self, Xb, yb):
        ''' Different from the loss because of the reg.
        Same if reg =0.0 '''
        return np.mean(self.loss_function(self.net.predict(Xb), yb))

    def get_loss_set(self, split='train'):
        ''' Return the MSE mean squared error for the whole set '''
        scores = []
        for Xb, yb in self.which_batch_iterator[split]:
            scores.append(self.get_loss(Xb, yb))
        return np.array(scores).mean()

    def get_score_set(self, split='train'):

        df = self.predict_yield(split)
        df = df[df['yield'] != -1]
        return np.sqrt(mean_squared_error(df['yield'], df['yield_pred']))

    def predict_yield(self, split):
        ''' The way we construct the data is a convolution.
        Therefore we have to unconvolver the batch by gp to be
        abble to reconstruct the original distribution.
        The net predict several value for each output which we
        average.

        Example.
        x = [1,2,3,4,5]
        y = [1,0,0,1,0]
        x_transform = [[1,2,3],[2,3,4],[3,4,5]]
        y_transform = [[1,0,0],[0,0,1],[0,1,0]]

        y_predict = [[0.2,0,0],[0.3,0.1,0.5],[0.4,0.3,0.4]]
        y_back_convolved = [[0.2,0,0,0,0],[0,0.3,0.1,0.5,0],[0,0,0.4,0.3,04]
        and next we mean on the first axis to have smth of shape 5 ; )
        '''

        df_pred = {}
        df = self.which_batch_iterator[split].df
        for gp, (X, y, p) in self.which_batch_iterator[split].stack_seqs.iteritems():
            mask = p[0].astype('int')
            ypred = self.net.predict(X)
            ypred_reshape = np.zeros(p[1])
            for k in range(ypred_reshape.shape[0]):
                ypred_reshape[k, mask[k, :]] = ypred[k, :]
            df_pred.update(
                dict(zip(df[df.group == gp].index, np.mean(ypred_reshape, axis=0))))
        df_pred = pd.DataFrame(
            df_pred.values(), index=df_pred.keys(), columns=['yield_pred'])
        return df.join(df_pred)

    def make_prediction(self):
        pred = []
        pred.append(self.predict_yield('train'))
        pred.append(self.predict_yield('val'))
        pred.append(self.predict_yield('test'))
        pred = reduce(lambda a, b: a.append(b), pred)
        pred = pred[pred['type'] == 'prediction']
        return pred

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

    def write_final_score(self):

        train = self.get_score_set(split='train')
        val = self.get_score_set(split='val')
        f = os.path.join(self.folder,
                         'train_%1.3f_val_%1.3f' % (train, val))
        with open(os.path.expanduser(f), 'wr+') as f:
            f.write('nice training')
            f.close()
