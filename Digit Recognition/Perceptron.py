__author__ = 'bdwalker'

import numpy as np
import Kernels


class Classifier:

    # Method that will make a prediction based on the kernel passed, the array of
    # mistakes made up to the current time step and the current observation.  It will
    # return either a 1 or a -1, indicating which class the current observation is
    # predicted to belong to.
    def classify(self, x, mistakes, kernel):
        mistake_y = mistakes[:, 0]
        mistake_x = mistakes[:, 1:]

        #print "Length of x: ", len(x)
        #print "Length of mistakes_x: ", len(mistake_x[0])

        c =  1 if (np.sum(mistake_y * np.apply_along_axis(kernel, 1, mistake_x, x))) > 0 else -1
        return c

    # Method to train the model.  It accepts as arguments and ndarray of training data,
    # an ndarray of their corresponding labels, the penalty constant, the kernel to use
    # on the training data and a boolean indicating if the average loss should be printed
    # during training (true if so, false otherwise). It will return an ndarray of all the
    # mistakes made during predictions that can be used for making predictions against
    # testing data.
    def _trainSVM(self, X, y, report, kernel):
        mistakes = np.zeros((1, len(X[0]) + 1))
        avgLoss = 0.0

        # loop over all samples
        for i in range(0, len(X)):
            x = X[i, :]
            prediction = self.classify(x, mistakes, kernel)

            # if prediction is incorrect, store it in mistakes array
            if prediction * y[i] <= 0:
                new_mistake = np.hstack((y[i], x))
                mistakes = np.row_stack((mistakes, new_mistake))
                avgLoss += 1

            # print average loss if report is True
            if (i + 1) % 100 == 0 and report:
                print avgLoss / i

        return mistakes

    # Method that accepts testing data as an ndarray and a trained model.  It will
    # make predictions for the test data with the trained model and return the predictions
    # in an ndarray.
    def testWithModel(self, x, kernel, mistakes):
        return np.apply_along_axis(self.classify, 1, x, mistakes, kernel)

    # Simple helper method to unpack everything in the model.
    def unpackModel(self, model):
        mistakes = model["weights"]
        print  mistakes[(3.0, 4.0)][1]
        return model["weights"]

    # Helper method to pack everything into a dictionary that will be needed
    # for testing with the trained model.  It will save all the mistakes needed
    # to make a prediction, the constant C needed for weighting and the kernel
    # used for training which should be identical to the one used for testing.
    def packModel(self, mistakes):
        print mistakes[(3.0, 4.0)][1]
        return {"weights": mistakes}

    # Trains the SVM model using the data passed in as x_data.  It will return
    # a trained model that can be used for making predictions on test data.  The
    # method accepts an integer representing the penalty to assign to the margin,
    # the kernel, if any, to be used to map the data to a higher dimension, a step size
    # eta and a boolean to indicate if the traverage loss should be printed during training.
    def trainModel(self, x_data, kernel=Kernels.defaultKernel(), report=False):

        # +1 for intercept
        self.weights = [0] * (x_data.shape[1] + 1)
        y = x_data[:, 0]

        # add a column of ones for the intercept
        X = np.hstack((np.ones((len(x_data), 1)), x_data[:, 1:]))

        mistakes = self._trainSVM(x_data[:, 1:], y, report, kernel)
        # Returns length 785
        return mistakes
