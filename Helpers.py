import random

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


def test_data(data, network):
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    totalacc = 0.0

    for words, labels in data:
        dy.renew_cg()
        num_words = len(words) - 2
        words_nums = np.array([tuple(word for word in words[i - 2:i + 3]) for i in range(2, num_words)])
        labels_num = np.array([label for label in labels[2:-2]])
        loss, prediction = network.forward(words_nums, labels_num)
        for i in range(0, prediction.size):
            if prediction[i] == "O" and labels_num[i] == "O":
                continue
            elif prediction[i] == labels_num[i]:
                correct += 1.0
                # print "Correct, Predicted", prediction[i], "Real:", labels_num[i]
                # else:
                #   print "Wrong, Predicted", prediction[i], "Real:", labels_num[i]

            totalacc += 1.0
        total += prediction.size
        total_loss += loss.value()
    acc = correct / totalacc
    total_loss /= total

    return total_loss, acc


def train_model(train, dev, network, trainer, num_iterations, save_file, droput1=0.0, droput2=0.0):
    dev_losses = list()
    dev_accs = list()
    best_acc = -np.inf
    for I in xrange(num_iterations):
        random.shuffle(train)
        random.shuffle(dev)
        total_loss = 0.0
        correct = 0.0
        total = 0.0
        totalacc = 0.0
        for words, labels in train:
            dy.renew_cg()
            num_words = len(words) - 2
            # Convert
            words_nums = np.array([tuple(word for word in words[i - 2:i + 3]) for i in range(2, num_words)])

            labels_num = np.array([label.strip("\n") for label in labels[2:-2]])
            loss, prediction = network.forward(words_nums, labels_num, droput1, droput2)
            for i in range(0, prediction.size):
                if prediction[i] == "O" and labels_num[i] == "O":
                    continue
                elif prediction[i] == labels_num[i]:
                    correct += 1.0
                totalacc += 1.0

            total += prediction.size
            total_loss += loss.value()
            loss.backward()
            trainer.update()
        dev_loss, dev_acc = test_data(dev, network)
        if dev_acc > best_acc:
            #network.save_model(save_file)
            best_acc = dev_acc
        dev_losses.append(dev_loss)
        dev_accs.append(dev_acc)

        print "Itertation:", I
        print "Training Loss:", total_loss / total
        print "Training Accuracy:", correct / totalacc
        print "Dev Loss:", dev_loss
        print "Dev Accuracy:", dev_acc

    return dev_losses, dev_accs


# Plots the result of the training.
def plot_results(history, title, ylabel, xlabel='Epoch'):
    plt.plot(history)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
