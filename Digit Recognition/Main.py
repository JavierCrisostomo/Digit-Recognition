import numpy as np
import SVM
import Test
import Train
import sys
import Kernels

output = None
train_input = None
test_input = None
train_data = None
test_data = None


def loadTrainData():
    global train_data, train_input, y
    train_data = np.loadtxt(train_input, delimiter=",", skiprows=1)


def loadTestData():
    global test_data, test_input
    test_data = np.loadtxt(test_input, delimiter=",", skiprows=1)


def main():
    if len(sys.argv) < 4:
        print "usage: train_file test_file output_directory"

    global train_input, test_input, output, y, test_data
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    output = sys.argv[3]

    loadTrainData()
    loadTestData()

    # TODO this is just for testing and will eventually be cleaned up

    #trainer = Train.Trainer()
    #trainer.processData(train_data, output_file="./model/parsed_train_data")
    #trainer.trainModel(report=True)
    svm = SVM.Classifier()
    tester = Test.TestModel()
    kernel = Kernels.exactPolyKernel(5)
    model = svm.trainModel(train_data, penalty=1, kernel=kernel, report=True)
    predictions = tester.predictWithModel(model, test_data)
    # error = tester.calculateLoss(predictions, test_data[:, 0])
    # print float(error) / len(test_data)

if __name__ == "__main__":
    main()
