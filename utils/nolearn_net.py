import numpy as np
import lasagne as nn
import theano
from nolearn.lasagne import NeuralNet as BaseNeuralNet
from time import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


class NeuralNet(BaseNeuralNet):

    def _check_good_input(self, X, y=None):
        pass

    def fit(self, epochs=None):
        self.initialize()

        try:
            self.train_loop(epochs=epochs)
        except KeyboardInterrupt:
            pass
        return self

    def partial_fit(self, classes=None):
        return self.fit(epochs=1)

    def train_loop(self, epochs=None):
        epochs = epochs or self.max_epochs

        on_batch_finished = self.on_batch_finished
        if not isinstance(on_batch_finished, (list, tuple)):
            on_batch_finished = [on_batch_finished]

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
        )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
        )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        while epoch < epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            if self.custom_scores:
                custom_scores = [[] for _ in self.custom_scores]
            else:
                custom_scores = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train:
                batch_train_loss = self.apply_batch_func(
                    self.train_iter_, Xb, yb)
                train_losses.append(batch_train_loss)

                for func in on_batch_finished:
                    func(self, self.train_history_)

            for Xb, yb in self.batch_iterator_test:
                batch_valid_loss, accuracy = self.apply_batch_func(
                    self.eval_iter_, Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

                if self.custom_scores:
                    y_prob = self.apply_batch_func(self.predict_iter_, Xb)
                    for custom_scorer, custom_score in zip(self.custom_scores, custom_scores):
                        custom_score.append(custom_scorer[1](yb, y_prob))

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if custom_scores:
                avg_custom_scores = np.mean(custom_scores, axis=1)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
            }
            if self.custom_scores:
                for index, custom_score in enumerate(self.custom_scores):
                    info[custom_score[0]] = avg_custom_scores[index]
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

    def get_output(self, layer):
        if isinstance(layer, basestring):
            layer = self.layers_[layer]

        fn_cache = getattr(self, '_get_output_fn_cache', None)
        if fn_cache is None:
            fn_cache = {}
            self._get_output_fn_cache = fn_cache

        if layer not in fn_cache:
            xs = self.layers_[0].input_var.type()
            get_activity = theano.function([xs], get_output(layer, xs))
            fn_cache[layer] = get_activity
        else:
            get_activity = fn_cache[layer]

        outputs = []
        for Xb, yb in self.batch_iterator_test():
            outputs.append(get_activity(Xb))
        return np.vstack(outputs)

    def predict(self, Xb):
        return self.apply_batch_func(self.predict_iter_, Xb)

    def predict_whole_set(self, split):
        probas = []
        which_batch_iterator = {'test': self.batch_iterator_test,
                                'train': self.batch_iterator_train}
        for Xb, yb in which_batch_iterator[split]:
            probas.append(self.predict(Xb, yb))
        return np.vstack(probas)

    def get_score(self, Xb, yb):
        score = mean_squared_error if self.regression else accuracy_score
        return float(np.sqrt(score(self.predict(Xb), yb)))

    def get_score_whole_set(self, split='test'):
        ''' Return the RMSE root mean squared error '''

        which_batch_iterator = {'test': self.batch_iterator_test,
                                'train': self.batch_iterator_train}
        scores = []
        for Xb, yb in which_batch_iterator[split]:
            scores.append(self.get_score(Xb, yb))
        return float(np.sqrt(np.array(scores).mean()))
