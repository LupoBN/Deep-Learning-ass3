from Helpers import *
import sys

EMBED_SIZE = 150
LAYERS = 1
DIMS = [64, 64]

if __name__ == '__main__':
    train, dev, network, trainer = prepare(sys.argv[1], sys.argv[2], sys.argv[4],
                                           EMBED_SIZE, LAYERS, DIMS)
    train_bi_lstm(train, dev, 5, trainer, network, sys.argv[3])