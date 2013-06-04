__author__ = 'bdwalker'

import numpy as np
import math
import Kernels


class Classifier:

    # Helper method to calculate the regularization to apply to the weight update.
    # Currently using the derivative of the l2 norm (Just the sums of the weights
    # multiplied by their label).
    # TODO  I am not confident this is correct
    def calculateRegularization(self, mistakes):
        product = np.sum(np.multiply(np.atleast_2d(mistakes[:, 0]).T, mistakes[:, 1:]))
        return product

    # Method that will make a prediction based on the kernel passed, the array of
    # mistakes made up to the current time step and the current observation.  It will
    # return either a 1 or a -1, indicating which class the current observation is
    # predicted to belong to.
    def predict(self, x, mistakes, C, eta, kernel):
        mistake_y = mistakes[:, 0]
        mistake_x = mistakes[:, 1:]
        weight = self.calculateRegularization(mistakes)

        # sign(C * sum_j(y * k(x_j, x_t)) - 2 w)
        prediction = np.sign(C * np.sum((2 - mistake_y) * np.apply_along_axis(
            kernel, 1, mistake_x, x)) - 2 * weight)
        return prediction

    # Method to train the model.  It accepts as arguments and ndarray of training data,
    # an ndarray of their corresponding labels, the penalty constant, the kernel to use
    # on the training data and a boolean indicating if the average loss should be printed
    # during training (true if so, false otherwise). It will return an ndarray of all the
    # mistakes made during predictions that can be used for making predictions against
    # testing data.
    def _trainSVM(self, X, y, C, eta, report, kernel):
        mistakes = np.zeros((1, len(X[0]) + 1))
        avgLoss = 0.0

        # loop over all samples
        for i in range(0, len(X)):
            x = X[i, :]
            prediction = self.predict(x, mistakes, C, eta, kernel)

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
    def testWithModel(self, test_data, model):
        weights, C, kernel = self.unpackModel(model)

        # add a column of ones for the intercept
        test_data = np.hstack((np.ones((len(test_data), 1)), test_data))
        return np.apply_along_axis(self.predict, 1, test_data, weights,
                                   C, 1, kernel)

    # Simple helper method to unpack everything in the model.
    def unpackModel(self, model):
        return model["weights"], model["C"], model["kernel"]

    # Helper method to pack everything into a dictionary that will be needed
    # for testing with the trained model.  It will save all the mistakes needed
    # to make a prediction, the constant C needed for weighting and the kernel
    # used for training which should be identical to the one used for testing.
    def packModel(self, weights, C, kernel):
        return {"weights": weights, "C": C, "kernel": kernel}

    # Trains the SVM model using the data passed in as x_data.  It will return
    # a trained model that can be used for making predictions on test data.  The
    # method accepts an integer representing the penalty to assign to the margin,
    # the kernel, if any, to be used to map the data to a higher dimension, a step size
    # eta and a boolean to indicate if the average loss should be printed during training.
    def trainModel(self, x_data, penalty=1, kernel=Kernels.defaultKernel(),
                   eta=1, report=False):

        y = x_data[:, 0]

        # add a column of ones for the intercept
        X = np.hstack((np.ones((len(x_data), 1)), x_data[:, 1:]))

        weights = self._trainSVM(X, y, penalty, eta, report, kernel)
        return self.packModel(weights, penalty, kernel)
