__author__ = 'bdwalker'
import pickle
import numpy as np
import SVM
import Kernels
import sys
import multiprocessing.pool as mp


class Trainer:
    processed_data = None
    classifiers = None

    def loadProcessedData(self, input_file):
        input = open(input_file, "r")
        self.processed_data = pickle.load(input)

    # Separate data into buckets, keyed by the label
    def processData(self, raw_data, output_file=None, input_file=None):
        if input_file is None:
            print "***** Separating Data Into Classes..."
            train_data = self.loadClasses(raw_data)

            # serialize
            if output_file is not None:
                print "***** Serializing Data to %s" % output_file
                output = open(output_file, "w+")
                pickle.dump(train_data, output)

        else:
            print "***** Loading Data From %s" % input_file
            self.loadProcessedData(input_file)

        print "***** Combining Classes for Training"
        self.processed_data = self.combineClasses(train_data)

        print "***** Processing Data Complete"

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
                    combined = np.vstack((class1, class2))

                    # rows need to be shuffled, otherwise not i.i.d
                    np.random.shuffle(combined)
                    pairings[(key1, key2)] = combined

        return pairings

    def trainModel(self, kernel=Kernels.defaultKernel, sigma=0, report=False, plot=False, output=None):
        if self.processed_data is None:
            print "Training data has not been loaded! \n " \
                  "Must load data with a call to processData or loadProcessedData."
            sys.exit(1)

        # go get some lunch, this is gonna take some time
        threads = dict()
        pool = mp.ThreadPool(processes=1)
        for couple in self.processed_data.keys():
            arg = (self.processed_data[couple], kernel, sigma, report, plot)
            result = pool.apply_async(self.trainSingleClass, arg)
            threads[couple] = result

        trained_model = dict()
        for thread in threads.keys():
            trained_model[thread] = threads[thread].get()

        #serialize trained model
        if output is not None:
            out = open(output, "w+")
            pickle.dump(trained_model, out)

    def trainSingleClass(self, data, kernel, sigma, report, plot):
        svm = SVM.SVM(x_data=data, kernel=kernel, regularization=sigma, report=report)
        return svm.runClassifier()









