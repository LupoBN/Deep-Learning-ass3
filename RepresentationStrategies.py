import dynet as dy


class SimpleEmbedding(object):
    def __init__(self, embedding_size, vocab_size, model, W2I):
        self._E = model.add_lookup_parameters((vocab_size, embedding_size))
        self._W2I = W2I

    def represent(self, sequence):
        vecs = [self._E[self._W2I[i]] if i in self._W2I else self._E[self._W2I["UUUNKKK"]]
                for i in sequence]
        return vecs


class SubWordEmbedding(SimpleEmbedding):
    def represent(self, sequence):
        words_repr = [self._E[self._W2I[i]] if i in self._W2I else self._E[self._W2I["UUUNKKK"]]
                      for i in sequence]
        pre_repr = [
            self._E[self._W2I["Pre-" + i[0:3]]] if "Pre-" + i[0:3] in self._W2I else self._E[self._W2I["Pre-UNK"]]
            for i in sequence]
        suf_repr = [
            self._E[self._W2I["Suf-" + i[-4:-1]]] if "Suf-" + i[-4:-1] in self._W2I else self._E[self._W2I["Suf-UNK"]]
            for i in sequence]
        vecs = [words_repr[i] + pre_repr[i] + suf_repr[i] for i in range(0, len(words_repr))]
        return vecs


class CharacterLSTM(object):
    def __init__(self, layers, embedding_size, num_of_letters, lstm_dim, model, C2I):
        self._E = model.add_lookup_parameters((num_of_letters, embedding_size))
        self._C2I = C2I
        self._builder = dy.VanillaLSTMBuilder(layers, embedding_size, lstm_dim, model)

    def represent(self, sequence):
        s = self._builder.initial_state()
        representation = list()
        for word in sequence:
            vecs = [self._E[self._C2I[i]] if i in self._C2I else self._E[self._C2I["C-UNK"]] for i in word]
            lstm_out = s.transduce(vecs)
            representation.append(lstm_out[-1])
        return representation


class CharacterAndEmbedding(object):
    def __init__(self, layers, letter_embedding, C2I, W2I, word_embedding, lstm_dim,
                 model, repr_dim):
        self._cLSTM = CharacterLSTM(layers, letter_embedding, len(C2I), lstm_dim, model, C2I)
        self._embedding = SimpleEmbedding(word_embedding, len(W2I), model, W2I)
        self._W = model.add_parameters((repr_dim, lstm_dim + word_embedding))
        self._b = model.add_parameters(repr_dim)

    def represent(self, sequence):
        W = dy.parameter(self._W)
        b = dy.parameter(self._b)
        char_representation = self._cLSTM.represent(sequence)
        word_represention = self._embedding.represent(sequence)
        linear = list()
        for i in range(0, len(sequence)):
            embedded = dy.concatenate([char_representation[i], word_represention[i]])
            result = (W * embedded) + b
            linear.append(result)
        return linear

