import dynet as dy
from gen_examples import *
from Utils import *


class RNNAcceptor(object):
    def __init__(self, layers, in_dim, lstm_dim, hid_dim, out_dim, model, vocab_size, W2I):
        self._builder = dy.LSTMBuilder(layers, in_dim, lstm_dim, model)
        self._W_hid = model.add_parameters((hid_dim, lstm_dim))
        self._b_hid = model.add_parameters(hid_dim)
        self._W_out = model.add_parameters((out_dim, hid_dim))
        self._b_out = model.add_parameters(out_dim)
        self._E = model.add_lookup_parameters((vocab_size, in_dim))
        self._W2I = W2I

    def __call__(self, sequence):
        lstm = self._builder.initial_state()
        W_hid = self._W_hid.expr()
        b_hid = self._b_hid.expr()
        W_out = self._W_out.expr()
        b_out = self._b_out.expr()
        vecs = [self._E[self._W2I[i]] for i in sequence]

        outputs = lstm.transduce(vecs)
        result = dy.logistic(W_out * dy.tanh(W_hid * outputs[-1] + b_hid) + b_out)
        return result
