from bilstmTrain import *

if __name__ == '__main__':
    train, dev, network, trainer = prepare(sys.argv[1], sys.argv[4], sys.argv[3],
                                           EMBED_SIZE, LAYERS, DIMS)
    network.load_model(sys.argv[2])
    separator = ' '
    if '-t' in sys.argv:
        separator = "\t"
    blind_write(dev, network, sys.argv[5], separator)
