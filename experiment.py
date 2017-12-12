import dynet as dy
from gen_examples import *
from Utils import *
from RNNAcceptor import RNNAcceptor
import time

VOCAB_SIZE = 13
EMBED_SIZE = 30


def train_acceptor(train, dev, num_of_iterations, trainer, acceptor):
    time0 = time.time()
    for epoch in range(num_of_iterations):
        np.random.shuffle(train)
        sum_of_losses = 0.0
        acc = 0.0
        for sequence, label in train:
            dy.renew_cg()  # new computation graph
            preds = acceptor(sequence)
            y = dy.scalarInput(label)
            loss = dy.binary_log_loss(preds, y)

            sum_of_losses += loss.value()
            vals = 0
            if preds.value() >= 0.5:
                vals = 1
            if vals == label:
                acc += 1.0
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
        dy.renew_cg()  # new computation graph

        preds = acceptor(sequence)
        vals = 0
        if preds.value() >= 0.5:
            vals = 1
        if vals == label:
            acc += 1.0
        y = dy.scalarInput(label)
        loss = dy.binary_log_loss(preds, y)
        sum_of_losses += loss.value()
    return acc / len(dev), sum_of_losses / len(dev)


if __name__ == '__main__':
    train = create_dataset(10000)
    dev = create_dataset(1000)
    W2I = get_abcd_mapping()

    m = dy.Model()
    trainer = dy.AdamTrainer(m)
    acceptor = RNNAcceptor(1, EMBED_SIZE, 2, 2, 1, m, VOCAB_SIZE, W2I)
    train_acceptor(train, dev, 2, trainer, acceptor)
