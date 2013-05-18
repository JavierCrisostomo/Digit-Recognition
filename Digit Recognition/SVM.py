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

    def __init__(self, x_data, regularization, kernel=Kernels.defaultKernel, report=False, dim=1):
        self.X = x_data
        self.delta = regularization
        self.report = report
        self.kernel = kernel
        self.dim = dim
        self.mistakes = np.zeros((1, len(x_data[0])))

    def calculateRegularization(self, x, dim=0):
        #result = .5 * math.sqrt(np.apply_along_axis(kernel, 1, x, x, dim))
        #print result
        #return result
        pass

    def predict(self, x):
        y = self.mistakes[:, 0]
        x_mistake = self.mistakes[:, 1::]
        regularize = 0 #self.calculateRegularization(kernel, x_mistake)
        prediction = np.sign(sum(y * np.apply_along_axis(self.kernel, 1, x_mistake, x, self.dim)) + regularize)
        return prediction

    def train_svm(self):
        avgLoss = 0.0
        for i in range(0, len(self.X)):
            x = self.X[i, :]
            prediction = self.predict(x[1:])
            if prediction * x[0] <= 0:
                self.mistakes = np.row_stack((self.mistakes, x))
                avgLoss += 1

            if (i + 1) % 100 == 0 and self.report:
                print avgLoss / i

    def runClassifier(self):
        self.train_svm()