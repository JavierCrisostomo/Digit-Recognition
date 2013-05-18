__author__ = 'bdwalker'
import pickle
import numpy as np
import Kernels
import sys
import threading


class Trainer:
    processed_data = None
    classifiers = None

    def loadProcessedData(self, input_file):
        input = open(input_file, "r")
        self.processed_data = pickle.load(input)
        print self.processed_data[(0, 1)]

    # Separate data into buckets, keyed by the label
    def processData(self, raw_data, output_file):
        train_data = self.loadClasses(raw_data)
        self.processed_data = self.combineClasses(train_data)

        # serialize
        output = open(output_file, "w+")
        pickle.dump(self.processed_data, output)

    def loadClasses(self, raw_data):
        train_data = dict()
        for row in raw_data:
            label = row[0]
            if label in train_data.keys():
                current = np.atleast_2d(train_data[label])
                row = np.atleast_2d(row)
                train_data[label] = np.vstack((current, row))
            else:
                train_data[label] = row

        return train_data

    def combineClasses(self, train_data):
        pairings = dict()
        for key1 in train_data.keys():
            class1 = train_data[key1]
            class1[:, 0] = 1
            for key2 in train_data.keys():
                if key2 > key1:
                    class2 = train_data[key2]
                    class2[:, 0] = -1
                    pairings[(key1, key2)] = np.vstack((class1, class2))
        return pairings

    def trainModel(self, kernel=Kernels.defaultKernel, sigma=0, report=False, plot=False, output=None):
        if self.processed_data is None:
            print "Training data has not been loaded! \n " \
                  "Must load data with a call to processData or loadProcessedData."
            sys.exit(1)

        threads = dict()
        # go get some lunch, this is gonna take some time
        for couple in self.processed_data.keys():
            arg = (self.processed_data[couple], kernel, sigma, report, plot)
            thread = threading.Thread(group=None, target=self.trainSingleClass, args=arg)
            thread.start()
            threads[couple] = thread

        trained_model = dict()
        for thread in threads.keys():
            trained_model[thread] = threads[thread].join()

        #serialize trained model
        if output is not None:
            out = open(output, "w+")
            pickle.dump(trained_model, out)

    def trainSingleClass(self, data, kernel, sigma, report, plot):
        pass









