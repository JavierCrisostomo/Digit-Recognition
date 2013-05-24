import numpy as np
import SVM
import KNN
import Test
import Train
import sys
import Kernels
from multiprocessing.pool import Pool
import KNNCrossValidation as cv


output = None
train_input = None
test_input = None
train_data = None
test_data = None
test_y = None
version = None


def loadTrainData():
    global train_data, train_input, y
    train_data = np.loadtxt(train_input, delimiter=",", skiprows=1)


def loadTestData():
    global test_data, test_input, test_y, version
    test_data = np.loadtxt(test_input, delimiter=",", skiprows=1)
    if version.upper() == "SVM":
        test_y = np.loadtxt("./data/knn_benchmark.csv")


def main():
    if len(sys.argv) < 5 or (sys.argv[4] != "KNN" and sys.argv[4] != "SVM"):
        print "usage: train_file test_file output_directory version"
        quit()

    global train_input, test_input, output, y, test_data, test_y, version
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    output = sys.argv[3]
    version = sys.argv[4]

    loadTrainData()
    loadTestData()

    # TODO this is just for testing and will eventually be cleaned up

    #trainer = Train.Trainer()
    #trainer.processData(train_data, output_file="./model/parsed_train_data")
    #trainer.trainModel(report=True)


    ####################################
    # SVM DRIVER
    # svm = SVM.Classifier()
    # tester = Test.TestModel()
    # kernel = Kernels.exactPolyKernel(5)
    # model = svm.trainModel(train_data, penalty=1, kernel=kernel, report=True)
    # predictions = tester.predictWithModel(model, test_data)
    ####################################

    ####################################
    # KNN DRIVER
    #
    # Cross validation

    k = cv.crossValidate([1,2,3,4,5,10,15,20], 10, train_data[0:10000])
    output = open("./data/output.txt", "w+")
    
    knn = KNN.KNN(train_data, k)
    classifications = list()
    results = list()
    for i in range(0, len(test_data)):
        knn = KNN.KNN(train_data, 1)
        x_row = test_data[i, :]
        results.append(knn.classifySample, x_row)



    result = sum([np.sign(x) for x in np.subtract(test_y, classifications)])
    print result
    output.close()
    ######################################

    #k = cv.crossValidate([1,2,3,4,5,10,15,20], train_data[0:1000,])


    # error = tester.calculateLoss(predictions, test_data[:, 0])
    # print float(error) / len(test_data)

if __name__ == "__main__":
    main()
