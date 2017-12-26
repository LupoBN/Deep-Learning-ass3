from Utils import *
from RNNAcceptor import RNNAcceptor
import sys
from Helpers import *
EMBED_SIZE = 30





if __name__ == '__main__':
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    train = read_file(sys.argv[1], read_examples_parser)
    dev = read_file(sys.argv[2], read_examples_parser)
    c = Counter()
    for sample, label in train:
        c.update(sample)
    W2I = {key: i for i, key in enumerate(c)}
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for t, l in train]))))}
    I2L = {L2I[key]: key for key in L2I}

    m = dy.ParameterCollection()
    trainer = dy.AdamTrainer(m)
    acceptor = RNNAcceptor(1, EMBED_SIZE, 2, 2, 2, len(W2I), W2I, L2I, I2L, m)
    train_acceptor(train, dev, 50, trainer, acceptor, 50)
