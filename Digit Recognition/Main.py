import numpy as np
import sys

output = None
train_input = None
test_input = None
y = None
train_data = None
test_data = None


def loadTrainData():
    global train_data, train_input, y
    train_data = np.loadtxt(open(train_input, "r"), delimiter=",", skiprows=1)
    y = train_data[:, 0]

    # delete label
    train_data = np.delete(train_data, 0, 1)


def loadTestData():
    global test_data, test_input
    test_data = np.loadtxt(open(test_input, "r"), delimiter=",", skiprows=1)


def main():
    if len(sys.argv) < 4:
        print "usage: train_file test_file output_directory"

    global train_input, test_input, output
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    output = sys.argv[3]

    loadTrainData()
    loadTestData()

if __name__ == "__main__":
    main()
