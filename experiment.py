import dynet as dy
from gen_examples import *
from Utils import *
from RNNAcceptor import RNNAcceptor
import sys
import time

EMBED_SIZE = 30


def train_acceptor(train, dev, num_of_iterations, trainer, acceptor):
    time0 = time.time()
    for epoch in range(num_of_iterations):
        np.random.shuffle(train)
        sum_of_losses = 0.0
        acc = 0.0

        for sequence, label in train:
            loss, prediction = acceptor.forward(sequence, label)
            sum_of_losses += loss.value()
            if prediction == label:
                acc += 1
            loss.backward()
            trainer.update()

        dev_acc, dev_loss = test_acceptor(dev, acceptor)

        print "Itertation:", epoch + 1
        print "Training accuracy:", acc / len(train)
        print "Training loss:", sum_of_losses / len(train)
        print "Test accuracy:", dev_acc
        print "Test loss:", dev_loss

    time1 = time.time()
    print "Finished training after", time1 - time0, " seconds"


def test_acceptor(dev, acceptor):
    np.random.shuffle(dev)
    acc = 0.0
    sum_of_losses = 0.0

    for sequence, label in dev:
        loss, prediction = acceptor.forward(sequence, label)
        sum_of_losses += loss.value()
        if prediction == label:
            acc += 1
    return acc / len(dev), sum_of_losses / len(dev)


if __name__ == '__main__':
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
    acceptor = RNNAcceptor(1, EMBED_SIZE, 128, 64, 2, len(W2I), W2I, L2I, I2L, m)
    train_acceptor(train, dev, 200, trainer, acceptor)
