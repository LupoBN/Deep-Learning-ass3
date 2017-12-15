import dynet as dy
from gen_examples import *
from Utils import *


class RNNAcceptor(object):
    def __init__(self, layers, in_dim, lstm_dim, hid_dim, out_dim, vocab_size, W2I, L2I, I2L, model):
        self._E = model.add_lookup_parameters((vocab_size, in_dim))

        self._W_hid = model.add_parameters((hid_dim, lstm_dim))
        self._b_hid = model.add_parameters(hid_dim)
        self._W_out = model.add_parameters((out_dim, hid_dim))
        self._b_out = model.add_parameters(out_dim)
        self._builder = dy.VanillaLSTMBuilder(layers, in_dim, lstm_dim, model)

        self._W2I = W2I
        self._L2I = L2I
        self._I2L = I2L


    def __call__(self, sequence):
        dy.renew_cg()  # new computation graph

        s = self._builder.initial_state()

        W_hid = dy.parameter(self._W_hid)
        b_hid = dy.parameter(self._b_hid)
        W_out = dy.parameter(self._W_out)
        b_out = dy.parameter(self._b_out)

        # embedd each char in the seq, and feed it to the LSTM one word at a time.
        for char in sequence:
            char_embedded = dy.lookup(self._E, self._W2I[char])

            s = s.add_input(char_embedded)

        lstm_out = s.output()

        result = dy.softmax((W_out * dy.tanh((W_hid * lstm_out) + b_hid)) + b_out)
        return result

    def forward(self, sequence, label):
        out = self(sequence)
        prediction = self._I2L[np.argmax(out.npvalue())]
        loss = -dy.log(dy.pick(out, self._L2I[label]))
        return loss, prediction
