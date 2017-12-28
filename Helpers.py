import time
import numpy as np
import matplotlib.pyplot as plt
from BiLSTM import BiLSTMNetwork
from Utils import *
from RepresentationStrategies import *
from matplotlib.font_manager import FontProperties


def blind_write(data, network, output_filename, separator):
    output_file = open(output_filename, 'w')

    for words, labels in data:
        dy.renew_cg()  # new computation graph
        predictions = network.predict(words)
        for i in range(1, len(predictions) - 1):
            output_file.write(words[i] + separator + predictions[i] + "\n")
        output_file.write("\n")
    output_file.close()


def train_bi_lstm(train, dev, num_of_iterations, trainer, acceptor, save_file):
    results = list()
    best_acc = 0.0

    for epoch in range(num_of_iterations):

        np.random.shuffle(train)

        correct = 0.0
        incorrect = 0.0
        sum_of_losses = 0.0
        counter = 0
        for sequence, label in train:
            if counter % 500 == 0:
                dev_acc, dev_loss = test_bi_lstm(dev, acceptor)
                results.append(dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc

                    acceptor.save_model(save_file)
                print "Test accuracy:", dev_acc
                print "Test loss:", dev_loss
            dy.renew_cg()  # new computation graph
            loss, prediction = acceptor.forward(sequence, label)
            sum_of_losses += loss.value()
            loss.backward()
            trainer.update()
            for i, pred in enumerate(prediction):
                # print pred
                if pred == "O" and label[i + 1] == "O":
                    continue
                if pred == label[i + 1]:
                    correct += 1.0
                else:
                    # print "Wrong, Predicted:", pred, "True Label:", label[i + 2]
                    incorrect += 1.0

            counter += 1
        if counter % 500 == 0:
            dev_acc, dev_loss = test_bi_lstm(dev, acceptor)
            results.append(dev_acc)
            if dev_acc > best_acc:
                best_acc = dev_acc

                acceptor.save_model(save_file)
            print "Test accuracy:", dev_acc
            print "Test loss:", dev_loss

        print "Itertation:", epoch + 1
        print "Training accuracy:", correct / (correct + incorrect)
        print "Training loss:", sum_of_losses / (correct + incorrect)
    return results


def test_bi_lstm(dev, acceptor):
    np.random.shuffle(dev)
    sum_of_losses = 0.0
    correct = 0.0
    incorrect = 0.0
    for sequence, label in dev:
        dy.renew_cg()  # new computation graph

        if len(sequence) <= 1:
            continue
        loss, prediction = acceptor.forward(sequence, label)
        sum_of_losses += loss.value()
        for i, pred in enumerate(prediction):
            # print pred
            if pred == "O" and label[i + 1] == "O":
                continue
            if pred == label[i + 1]:
                # print "Correct, Predicted:", pred, "True Label:", label[i + 2]
                correct += 1.0
            else:
                # print "Wrong, Predicted:", pred, "True Label:", label[i + 2]

                incorrect += 1.0
    return correct / (correct + incorrect), sum_of_losses / (correct + incorrect)


def train_acceptor(train, dev, num_of_iterations, trainer, acceptor, batch_size):
    time0 = time.time()

    for epoch in range(num_of_iterations):
        counter = 0
        np.random.shuffle(train)
        mini_batches = [train[i:i + batch_size] for i in range(0, len(train), batch_size)]

        acc = 0.0
        sum_of_losses = 0.0

        dy.renew_cg()  # new computation graph
        for mini_batch in mini_batches:
            losses = []
            for sequence, label in mini_batch:
                loss, prediction = acceptor.forward(sequence, label)
                losses.append(loss)
                if prediction == label:
                    acc += 1.0

            batch_loss = dy.esum(losses)
            sum_of_losses += batch_loss.value()
            batch_loss.backward()
            trainer.update()
            counter += 1
        dev_acc, dev_loss = test_acceptor(dev, acceptor, batch_size)
        print "Itertation:", epoch + 1
        print "Training accuracy:", acc / len(train)
        print "Training loss:", sum_of_losses / len(train)
        print "Test accuracy:", dev_acc
        print "Test loss:", dev_loss

    time1 = time.time()
    print "Finished training after", time1 - time0, " seconds"


def test_acceptor(dev, acceptor, batch_size):
    np.random.shuffle(dev)
    mini_batches = [dev[i:i + batch_size] for i in range(0, len(dev), batch_size)]
    sum_of_losses = 0.0
    acc = 0.0
    dy.renew_cg()  # new computation graph
    for mini_batch in mini_batches:
        losses = []

        for sequence, label in mini_batch:
            loss, prediction = acceptor.forward(sequence, label)
            losses.append(loss)
            if prediction == label:
                acc += 1.0
        batch_loss = dy.esum(losses)
        sum_of_losses += batch_loss.value()  # this calls forward on the batch
    return acc / len(dev), sum_of_losses / len(dev)


# Plots the result of the training.
def plot_results(history, title, ylabel, xlabel='Sentences Seen / 100'):
    plt.title(title)
    plt.ylabel(ylabel)
    x = [i * 5 for i in range(0, len(history["1"]))]
    plt.plot(x, history['1'])
    plt.plot(x, history['2'])
    plt.plot(x, history['3'])
    plt.plot(x, history['4'])
    plt.xlabel(xlabel)
    plt.legend(['Embedding', 'Character Level LSTM', 'Embedding+Sub Words', 'Embedding + Character Level LSTM'],
               loc="lower right")

    plt.show()


def get_representation(m, W2I, train, arg, embed_size):
    if arg == "1":
        representor = SimpleEmbedding(embed_size, len(W2I), m, W2I)
    elif arg == "2":
        C2I = get_C2I(train)
        representor = CharacterLSTM(1, 10, len(C2I), embed_size, m, C2I)
    elif arg == "3":
        sub_map = sub_words_mapping(train, len(W2I))
        W2I.update(sub_map)
        representor = SubWordEmbedding(embed_size, len(W2I), m, W2I)
    elif arg == "4":
        C2I = get_C2I(train)
        representor = CharacterAndEmbedding(1, 10, C2I, W2I, embed_size, embed_size, m, embed_size)
    return representor


def prepare(repr, train_file, dev_file, embedding_size, layers, dims):
    train, labels = read_file(train_file, parse_tag_reading)
    dev, dev_labels = read_file(dev_file, parse_tag_reading)

    most = 40000
    T2I = create_mapping(labels, ignore_elements={"Start-", "End-"})
    if len(T2I) < 40:
        most = 20000
    W2I = create_mapping(train, most_to_take=most)
    I2T = [key for key, value in sorted(T2I.iteritems(), key=lambda (k, v): (v, k))]
    m = dy.ParameterCollection()
    representor = get_representation(m, W2I, train, repr, embedding_size)
    network = BiLSTMNetwork(layers, embedding_size, dims[0], dims[1], len(T2I), representor, T2I, I2T, m,
                            dy.SimpleRNNBuilder)
    trainer = dy.AdamTrainer(m)
    return zip(train, labels), zip(dev, dev_labels), network, trainer
