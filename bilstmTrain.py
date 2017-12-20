from Utils import *
from Helpers import *
from BiLSTM import BiLSTMNetwork
from RepresentationStrategies import *
import sys

EMBED_SIZE = 50

if __name__ == '__main__':
    #dyparams = dy.DynetParams()
    #dyparams.set_autobatch(True)
    representation_strategy = sys.argv[1]
    train, labels = read_file(sys.argv[2], parse_tag_reading)
    dev, dev_labels = read_file(sys.argv[3], parse_tag_reading)


    most = 40000
    T2I = create_mapping(labels, ignore_elements={"Start-", "End-"})
    if len(T2I) < 40:
        most = 20000
    W2I = create_mapping(train, most_to_take=most)
    I2T = [key for key, value in sorted(T2I.iteritems(), key=lambda (k, v): (v, k))]
    train = zip(train, labels)
    dev = zip(dev, dev_labels)

    m = dy.ParameterCollection()
    if sys.argv[1] == "1":
        representor = SimpleEmbedding(EMBED_SIZE, len(W2I), m, W2I)
    elif sys.argv[1] == "2":
        c = Counter()
        for key in W2I:
            c.update(key)
        C2I = {char: i for i, char in enumerate(c)}
        representor = CharacterLSTM(1, 10, len(C2I), EMBED_SIZE, m, C2I)
    elif sys.argv[1] == "3":
        sub_map = sub_words_mapping(train, len(W2I))
        W2I.update(sub_map)
        representor = SubWordEmbedding(EMBED_SIZE, len(W2I), m, W2I)
    elif sys.argv[1] == "4":
        c = Counter()
        for key in W2I:
            c.update(key)
        C2I = {char: i for i, char in enumerate(c)}
        representor = CharacterAndEmbedding(1, 10, C2I, W2I, EMBED_SIZE, EMBED_SIZE, m, EMBED_SIZE)




    # create network
    network = BiLSTMNetwork(1, EMBED_SIZE, 32, 32, len(T2I), representor, T2I, I2T, m, dy.SimpleRNNBuilder)
    # create trainer
    trainer = dy.AdamTrainer(m)
    train_bi_lstm(train, dev, 50, trainer, network, 1)
