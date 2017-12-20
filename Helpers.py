import time
import dynet as dy
import numpy as np
import matplotlib.pyplot as plt


def blind_write(data, network, output_filename, separator):
    output_file = open(output_filename, 'w')

    for words, labels in data:
        num_words = len(words) - 2
        words_nums = np.array([tuple(word for word in words[i - 2:i + 3]) for i in range(2, num_words)])
        prediction = network.predict(words_nums)
        for i in range(0, prediction.size):
            output_file.write(words_nums[i][2] + separator + prediction[i] + "\n")
        output_file.write("\n")
    output_file.close()


def train_bi_lstm(train, dev, num_of_iterations, trainer, acceptor, batch_size, history=200000):
    for epoch in range(num_of_iterations):
        np.random.shuffle(train)

        correct = 0.0
        incorrect = 0.0
        sum_of_losses = 0.0
        counter = 1
        for sequence, label in train:
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
            if counter % 500 == 0:
                pass
            counter += 1
        print "Itertation:", epoch + 1
        print "Training accuracy:", correct / (correct + incorrect)
        print "Training loss:", sum_of_losses / (correct + incorrect)
        dev_acc, dev_loss = test_bi_lstm(dev, acceptor, batch_size, history)
        print "Test accuracy:", dev_acc
        print "Test loss:", dev_loss


def test_bi_lstm(dev, acceptor, batch_size, history):
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
def plot_results(history, title, ylabel, xlabel='Epoch'):
    plt.plot(history)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
