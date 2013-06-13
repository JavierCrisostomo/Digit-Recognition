import KNN
import Train
import Test
import sys
import Kernels
import KNNCrossValidation as cv
import numpy as np
import Perceptron
import pickle
import PerceptronCrossValidation

def loadTrainData(input_file):
    return np.loadtxt(input_file, delimiter=",", skiprows=1)

def loadTestData(test_file, true_label="./uci_data/test/Y_test.txt"):
    test_data = np.loadtxt(test_file)
    test_y = None
    if true_label is not None:
        print true_label
        test_y = np.loadtxt(true_label)
    return test_data, test_y

def loadUCITrainData(input_x, input_y):
    train_x = np.loadtxt(input_x)
    train_y = np.atleast_2d(np.loadtxt(input_y)).T
    train_data = np.hstack((train_y, train_x))
    return train_data


def runKNNCV(input_file, output, folds):
    train_data = loadTrainData(input_file)
    k = cv.crossValidation([1,2,3,4,5,10,15,20], folds, train_data)

    if output is not None:
        out = open(output, "w")
        out.write(str(k))
        out.close()

def runKNN(train_file, test_file, output_file, k, weighted):
    train_data = loadTrainData(train_file)
    test_data, test_y = loadTestData(test_file, None)
    test_data = test_data[:]
    output = open(output_file, "w")

    knn = KNN.Classifier(train_data, k, weighted)
    results = list()

    count = 0
    for i in range(0, test_data.shape[0]):
        x_row = test_data[i, :]
        classified = knn.classifySample(x_row)
        if count % 1000 == 0:
            print count
        count += 1
        results.append(str(classified))

    output.writelines(results)
    output.close()

    print results

def trainPerceptron(train_file, kernel, output_file, dim, train_y = None):
    trainer = Train.Trainer()
    trainer.processData(loadUCITrainData(train_file, "./uci_data/train/Y_train.txt"))
    realKernel = None
    if kernel == "poly":
        realKernel = Kernels.exactPolyKernel(dim)
    elif kernel == "gaus":
        realKernel = Kernels.exponentialKernel(dim)
    trainer.trainModel(realKernel, True, False, output_file)

# Loads the test data and model, unpacks the model and 
# initializes the testing framework.
def runPerceptron(test_file, kernel, output_file, dim, model_file):
    testData, testY = loadTestData(test_file)
    tester = Test.TestModel()
    perceptron = Perceptron.Classifier()
    realKernel = None
    if kernel == "poly":
        realKernel = Kernels.exactPolyKernel(dim)
    elif kernel == "gaus":
        realKernel = Kernels.exponentialKernel(dim)

    infile = open(model_file, "r")
    model = pickle.load(infile)
    model = perceptron.unpackModel(model)
    classifications = tester.predictWithModel(model, testData, realKernel, True)
    output = open(output_file, "w+")
    output.writelines(classifications)

    print calculateError(classifications)


def calculateError(classifications):
    label = np.loadtxt("./uci_data/test/Y_test.txt")
    error = 0.0
    for i in range(len(classifications)):
        if int(classifications[i]) != (label[i]):
            error += 1.0
    return error / len(classifications)

def perceptronCV(data_file):
    PerceptronCrossValidation.crossValidate(loadUCITrainData(data_file, "./uci_data/train/Y_train.txt"))


def printUsage():

    print "usage: \n" \
          "KNN function train_file output_directory [test_file] [test_y] [k_value] [number_of_folds] [weight=true/false]\n" \
          "     function = cv (cross validate), test, weighted \n" \
          "     output_direct = location to write results\n" \
          "     test_file (optional if function = cv)\n" \
          "     test_y = results file for test (optional if function = cv)" \
          "     k_value (optional if function = cv)\n" \
          "     number_of_folds = folds to use for cross validation (optional if function = test)\n\n" \
          "     weight true/false if we want to run with weights" \
          "" \
          "SVM function train_file output_directory [test_file] [kernel] [number_of_folds]\n" \
          "     function = cv (cross validate), train, test\n" \
          "     train_file = data to train the classifier with\n" \
          "     output_directory = location to write results\n" \
          "     test_file (optional if function = cv, train\n" \
          "     kernel = kernel used with SVM (optional if function = cv)\n" \
          "     number_of_folds = folds to use for cross validation (optional if function = train, test) \n\n" \
          "Perceptron function train_file output_directory [test_file] [kernel] [number_of_folds] \n" \
          "     function = cv (cross validate), train, test\n" \
          "     train_file = data to train the classifier with\n" \
          "     output_directory = location to write results\n" \
          "     test_file (optional if function = cv, train\n" \
          "     kernel = kernel used with SVM (optional if function = cv)\n" \
          "     number_of_folds = folds to use for cross validation (optional if function = train, test) \n"
    sys.exit(1)

def main():
    if len(sys.argv) < 5:
        printUsage()

    method = sys.argv[1].lower()
    func = sys.argv[2].lower()
    train_file = sys.argv[3]
    output_file = sys.argv[4]

    try:
        if method == "svm":
            if func == "train":
                pass
            elif func == "test":
                pass
            elif func == "cv":
                pass
            else:
                printUsage()

        elif method == "knn":
            if func == "train":
                printUsage()
            elif func == "test":
                test_file = sys.argv[5]
                k = int(sys.argv[6])
                weighted = bool(sys.argv[7])
                runKNN(train_file, test_file, output_file, k, weighted)
            elif func == "cv":
                num_folds = int(sys.argv[5])
                runKNNCV(train_file, output_file, num_folds)
            else:
                printUsage()
        elif method == "perceptron":
            if func == "train":
                kernel = sys.argv[5]
                dim = int(sys.argv[6])
                trainPerceptron(train_file, kernel, output_file, dim)
            elif func == "test":
                kernel = sys.argv[5]
                dim = int(sys.argv[6])
                model_file = sys.argv[7]
                runPerceptron(train_file, kernel, output_file, dim, model_file)
            elif func == "cv":
                perceptronCV(train_file)
            else:
                printUsage()
        else:
            printUsage()
    except IndexError as e:
        printUsage()


if __name__ == "__main__":
    main()
