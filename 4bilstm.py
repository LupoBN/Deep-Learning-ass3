from bilstmTrain import *

if __name__ == '__main__':

    results = {}
    for i in range(1, 5):
        train, dev, network, trainer = prepare(str(i), sys.argv[2], sys.argv[4], EMBED_SIZE, LAYERS, DIMS)
        results[str(i)] = (train_bi_lstm(train, dev, 5, trainer, network, sys.argv[3]))
    plot_results(results, "Models Accuracy", "Accuracy")
