import KNN
import Train
import sys
import Kernels
import KNNCrossValidation as cv
import numpy as np


def loadTrainData(input_file):
    return np.loadtxt(input_file, delimiter=",", skiprows=1)


def loadTestData(test_file, true_label=None):
    print test_file
    #test_data = np.loadtxt(test_file, delimiter=",", skiprows=0)
    test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
    test_y = None
    if true_label is not None:
        print true_label
        test_y = np.loadtxt(true_label, delimiter=",", skiprows=1)
    return test_data, test_y

def runSVM():
    trainer = Train.Trainer()
    trainer.processData(train_data, None, "./trained_model.csv")
    kernel = Kernels.exactPolyKernel(5)
    trainer.trainModel(kernel, True, False, "./trained_model.csv")
    pass

def runKNNCV(input_file, output, folds):
    train_data = loadTrainData(input_file)
    k = cv.KNNCrossValidation([1, 2, 3, 4, 5, 10, 15, 20], folds, train_data)

    if output is not None:
        out = open(output, "w")
        out.write(str(k))
        out.close()

def runKNN(train_file, test_file, true_label, output_file, k, weighted):
    train_data = loadTrainData(train_file)
    test_data, test_y = loadTestData(test_file, true_label)
    test_data = test_data[:]
    #test_data = test_data[:, 1:]
    output = open(output_file, "w")

    knn = KNN.Classifier(train_data, k, weighted)
    results = list()

    for i in range(0, test_data.shape[0]):
        x_row = test_data[i, :]
        classified = knn.classifySample(x_row)
        results.append(str(classified))

    output.writelines(results)
    output.close()

    print results
    print true_label


def printUsage():

    print "usage: \n" \
          "KNN function train_file output_directory [test_file] [test_y] [k_value] [number_of_folds]\n" \
          "     function = cv (cross validate), test, weighted \n" \
          "     output_direct = location to write results\n" \
          "     test_file (optional if function = cv)\n" \
          "     test_y = results file for test (optional if function = cv)" \
          "     k_value (optional if function = cv)\n" \
          "     number_of_folds = folds to use for cross validation (optional if function = test)\n\n" \
          "" \
          "SVM function train_file output_directory [test_file] [kernel] [number_of_folds]\n" \
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
                label_file = sys.argv[6]
                k = int(sys.argv[7])
                runKNN(train_file, test_file, label_file, output_file, k, False)
            elif func == "weighted":
                test_file = sys.argv[5]
                label_file = sys.argv[6]
                k = int(sys.argv[7])
                runKNN(train_file, test_file, label_file, output_file, k, True)
            elif func == "cv":
                num_folds = int(sys.argv[5])
                runKNNCV(train_file, output_file, num_folds)
            else:
                printUsage()

        else:
            printUsage()
    except IndexError as e:
        printUsage()


if __name__ == "__main__":
    main()
