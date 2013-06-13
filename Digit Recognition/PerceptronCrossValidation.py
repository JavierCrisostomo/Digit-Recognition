import numpy as np
from time import time
import multiprocessing as mp
import Perceptron
import Kernels
import pickle
import Train
import Test


def crossValidate(train_data):
    _trainModel(train_data)

def _trainModel(train_data):
    dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25]
    trainer = Train.Trainer()
    test_size = int(train_data.shape[0] / 10)
    folds = 10
    pool = mp.Pool(processes=4)
    results = list()

    output = open("./results/uci_cv_results_perceptron.txt", "w+")
    for dim in dimensions:
        args = (test_size, trainer, folds, train_data, dim)
        results.append(pool.apply_async(_crossValidate, args))

    pool.close()
    pool.join()

    for result in results:
        error, test_time, dim = result.get()
        output.write("Testing Time: %f" % test_time + " minutes \n")
        output.write("Polynomial Kernel (dim=" + str(dim) + "):" + str(error / folds) + "\n\n")

    output.close()

def _crossValidate(test_size, trainer, folds, train_data, dim):
    start = 0
    error = 0
    kernel = Kernels.polyKernel(dim)
    start_time = time()
    for i in range(folds):
        split_test = train_data[range(start, start + test_size), 1:]
        split_label = train_data[range(start, start + test_size), 0]
        split_train = np.delete(train_data, range(start, start + test_size), axis=0)
        print "Test Size ", split_test.shape
        trainer.processData(split_train)
        model_file = "./Model/uci_perceptron_model_poly_" + str(dim)
        trainer.trainModel(kernel, True, False, model_file)
        error += _testModel(split_test, split_label, model_file, kernel)
        start += test_size
    end_time = time()
    return error, (end_time - start_time) / 60.0, dim

def _error(classification, labels):
    n = labels.shape[0]
    count = 0
    for i in range(len(classification)):
        if int(classification[i]) != int(labels[i]):
            count += 1

    return float(count)/n


def _testModel(test_data, label, model_file, kernel):
    percep = Perceptron.Classifier()
    tester = Test.TestModel()
    inputf = open(model_file, "r")
    inFile = pickle.load(inputf)
    model = percep.unpackModel(inFile)
    classifications = tester.predictWithModel(model, test_data, kernel, True)
    return _error(classifications, label)

