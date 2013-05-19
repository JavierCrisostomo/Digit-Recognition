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

    #trainer = Train.Trainer()
    #trainer.processData(train_data, output_file="./model/parsed_train_data")
    #trainer.trainModel(report=True)
    svm = SVM.SVM(train_data, regularization=0, kernel=Kernels.polyKernel, dim=5, report=True)
    tester = Test.TestModel()
    model = svm.trainModel()
    predictions = tester.predictWithModel(model, test_data)
    error = tester.calculateLoss(predictions, test_data[:, 0])
    print float(error) / len(test_data)

if __name__ == "__main__":
    main()
