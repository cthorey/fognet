import lasagne
import importlib
from hook import write_final_score
from data_utils import *
from nolearn_net import NeuralNet
from checkpoints import *
from helper import *
from preprocessing import *


class Model(object):

    def __init__(self, config, mode='train', hp=['lr', 'reg', 'hiddens']):
        self.mode = mode
        self.conf = config
        for key, val in config.iteritems():
            setattr(self, key, val)
        self.hp = {f: config[f] for f in hp}
        self.init_data()
        self.on_epoch_finished = []
        self.init_checkpoints()
        self.init_model(mode=mode)

    def init_data(self):
        ################################################################
        # Load the preprocessing
        print 'Loading the prepro pipeline : %s' % self.pipe_list
        pipeline = build_pipeline(self.pipe_list, self.pipe_kwargs)

        ################################################################
        # Load the iterator
        # Initialize the batchiterator
        print 'Loading data iterator using : %s' % self.build_ite
        self.nb_features, self.batch_ite_train, self.batch_ite_val, self.batch_ite_test, self.batch_ite_pred = load_data(
            name=self.name,
            feats=self.feats,
            build_ite=self.build_ite,
            pipeline=pipeline,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            stride=self.stride)

    def init_architecture(self):
        ################################################################
        # Build the architecture
        model = importlib.import_module(
            'model_defs.%s' % self.model)
        builder = getattr(model, self.architecture)
        self.architecture = builder(D=self.nb_features, H=self.hiddens,
                                    grad_clip=self.grad_clip)

    def init_checkpoints(self):
        # Model checkpoints
        name = '_'.join([f + '_' + str(g) for f, g in self.hp.iteritems()])
        self.model_name = name
        self.folder = os.path.join(self.root, name)
        output_exists = os.path.isdir(self.folder)
        if output_exists and not self.overwrite:
            print 'Model output exists. Use --overwrite'
            sys.exit(1)
        elif not output_exists:
            os.mkdir(self.folder)

        self.model_fname = os.path.join(self.folder, 'model.pkl')
        self.model_history_fname = os.path.join(
            self.folder, 'model_history.pkl')
        self.model_graph_fname = os.path.join(self.folder, 'model_history.png')

        save_weights = SaveWeights(
            self.model_fname, only_best=True, pickle=False)
        save_training_history = SaveTrainingHistory(self.model_history_fname)
        plot_training_history = PlotTrainingHistory(self.model_graph_fname)
        early_stopping = EarlyStopping(patience=self.patience)

        self.on_epoch_finished = [
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping
        ]

    def init_model(self, mode='train'):
        ''' Main function that build the model given everythin '''

        print 'Build the architecture: %s, %s' % (self.model, self.architecture)
        self.init_architecture()

        if mode == 'train':
            print 'Set up the checkpoints'
            self.init_checkpoints()

        ################################################################
        # Initialize solver
        print 'Initialize the network '
        self.net = NeuralNet(
            layers=self.architecture,
            regression=True,
            objective_loss_function=getattr(
                lasagne.objectives, self.loss_function),
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
            unwanted = ['architecture', 'batch_ite_pred', 'batch_ite_test',
                        'batch_ite_val', 'batch_ite_train', 'conf', 'net', 'on_epoch_finished']
            config = {k: v for k, v in props(
                self).iteritems() if k not in unwanted}
            self.conf = config
            dump_conf_file(config, self.folder)

    def train(self):
        ''' The function that allow training giving a specific model given
        a config file '''

        if self.mode != 'train':
            raise ValueError('run in training mode : mode=train')

        ################################################################
        # Reload the weights if we go from an older mode
        if self.continue_training:
            print 'Loading model params from %s' % self.model_fname
            self.net.load_params_from(self.model_fname)
            with open(self.model_history_fname) as f:
                net.train_history_ = pickle.load(f)

        ###########################################################
        # Fitting
        self.net.fit(epochs=self.nb_epochs)

        ###############################################################
        # Final score
        print 'Loading best param'
        self.net.load_params_from(self.model_fname)

        print 'Evaluating on test set'
        print self.net.get_score_whole_set(split='test')

        ################################################################
        # Write final score in the folder as a name of txt file
        write_final_score(config, self.net)

        ################################################################
        # Predict the yield for the whole prediction set
        print 'Run the prediction'
        final_pred = prediction(net, batch_ite_pred)
        make_submission(config, final_pred)

    def predict_unseen_data(self):
        ''' Given a net and an iterator,
        return the unique set of yield  prediction along
        with the date
        '''
        final_pred = {}
        df_pred = self.batch_iterator_pred.df
        for gp, X, p in self.batch_iterator_pred:
            mask = p[0].astype('int')
            ypred = self.net.predict(X)
            ypred_reshape = np.zeros(p[1])
            for k in range(ypred_reshape.shape[0]):
                ypred_reshape[k, mask[k, :]] = ypred[k, :]
            final_pred.update(
                dict(zip(df_pred[df_pred.group == gp].index, np.mean(ypred_reshape, axis=0))))
        final_pred = pd.DataFrame(
            final_pred.values(), index=final_pred.keys(), columns=['yield_pred'])
        return final_pred

    def make_submission(self, df):
        ''' Given a dataframe, make the prediction '''

        output_fname = os.path.join(
            self.folder, 'submissions_%s.csv' % get_current_datetime())
        print 'Will write output to %s' % output_fname

        ################################################################
        # Merge and produce  the submission file
        submission_df = load_raw_data()['submission_format']
        final_pred_format = submission_df.join(df, how='left')
        submission_df['yield'] = final_pred_format['yield_pred']

        ################################################################
        # Remove value below zero !
        submission_df[submission_df['yield'] < 0.0] = 0

        ################################################################
        # Store to a txt file
        submission_df.to_csv(output_fname)
