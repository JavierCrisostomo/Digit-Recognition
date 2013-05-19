__author__ = 'bdwalker'

import numpy as np
import math
import Kernels


class SVM:
    mistakes = None
    delta = None
    kernel = None
    dim = 1
    report = None
    X = None

    def __init__(self):
        pass

    def __init__(self, x_data=None, regularization=0, kernel=Kernels.defaultKernel, report=False, dim=1):
        self.X = x_data
        self.delta = regularization
        self.report = report
        self.kernel = kernel
        self.dim = dim

    def calculateRegularization(self):
        result = self.delta * math.sqrt(np.sum(self.kernel(self.mistakes, self.mistakes, self.dim)))
        return result

    def predict(self, x):
        y = self.mistakes[:, 0]
        x_mistake = self.mistakes[:, 1::]
        regularize = self.calculateRegularization()
        prediction = np.sign(np.sum(y * np.apply_along_axis(self.kernel, 1, x_mistake, x, self.dim)) + regularize)
        return prediction

    def trainSVM(self):
        avgLoss = 0.0
        for i in range(0, len(self.X)):
            x = self.X[i, :]
            prediction = self.predict(x[1:])
            if prediction * x[0] <= 0:
                self.mistakes = np.row_stack((self.mistakes, x))
                avgLoss += 1
            if (i + 1) % 100 == 0 and self.report:
                print avgLoss / i

    def testWithModel(self, test_data, model=None):
        if model is not None:
            self.mistakes = model

        return np.apply_along_axis(self.predict, 1, test_data)

    def trainModel(self):
        self.mistakes = np.zeros((1, len(self.X[0])))
        self.trainSVM()
        return self.mistakes