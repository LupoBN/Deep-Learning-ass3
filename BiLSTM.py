import dynet as dy
import numpy as np


class BiLSTMNetwork(object):
    def __init__(self, layers, in_dim, lstm_dim, second_lstm_dim, out_dim,
                 representor, L2I, I2L, model, builder=dy.VanillaLSTMBuilder):
        self._model = model
        # the embedding paramaters

        self._fwd_RNN_first = builder(layers, in_dim, lstm_dim, model)
        self._bwd_RNN_first = builder(layers, in_dim, lstm_dim, model)

        self._fwd_RNN_second = builder(layers, lstm_dim * 2, second_lstm_dim, model)
        self._bwd_RNN_second = builder(layers, lstm_dim * 2, second_lstm_dim, model)

        self._W_out = model.add_parameters((out_dim, second_lstm_dim * 2))
        self._b_out = model.add_parameters(out_dim)
        self._representor = representor
        self._L2I = L2I
        self._I2L = I2L

    def __call__(self, sequence):
        W_out = dy.parameter(self._W_out)
        b_out = dy.parameter(self._b_out)
        vecs = self._representor.represent(sequence)

        first_fwd = self._fwd_RNN_first.initial_state()
        first_bwd = self._bwd_RNN_first.initial_state()
        second_fwd = self._fwd_RNN_second.initial_state()
        second_bwd = self._fwd_RNN_second.initial_state()

        rnn_fwd_first = first_fwd.transduce(vecs)
        rnn_bwd_first = (first_bwd.transduce(vecs[::-1]))[::-1]

        first_layer_fwd = [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(rnn_fwd_first, rnn_bwd_first)]
        first_layer_bwd = first_layer_fwd[::-1]
        rnn_fwd_second = second_fwd.transduce(first_layer_fwd)
        rnn_bwd_second = (second_bwd.transduce(first_layer_bwd))[::-1]
        lstm_out = [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(rnn_fwd_second, rnn_bwd_second)]

        results = list()
        for i in range(len(lstm_out)):
            result = dy.softmax((W_out * lstm_out[i]) + b_out)
            results.append(result)
        return results

    def forward(self, sequence, label):
        out = self(sequence)
        predictions = list()
        losses = list()
        for single_output, single_label in zip(out, label):
            if single_label == "Start-" or single_label == "End-":
                continue
            prediction = self._I2L[np.argmax(single_output.npvalue())]
            predictions.append(prediction)
            loss = -dy.log(dy.pick(single_output, self._L2I[single_label]))
            losses.append(loss)
        loss = dy.esum(losses)
        return loss, predictions

    def predict(self, sequence):
        out = self(sequence)
        predictions = list()
        for single_output in out:
            prediction = self._I2L[np.argmax(single_output.npvalue())]
            predictions.append(prediction)
        return predictions




    def save_model(self, model_file):
        self._model.save(model_file)

    def load_model(self, model_file):
        self._model.populate(model_file)
