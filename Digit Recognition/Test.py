__author__ = 'bdwalker'

import KNN
import SVM
import Perceptron


class TestModel:

    def predictWithModel(self, model, test_data, report=False):
        perceptron = Perceptron.Classifier()
        predictions = perceptron.testWithModel(test_data, model)
        return predictions

    def calculateLoss(self, predictions, labels):
        error = 0
        for i in range(0, len(predictions)):
            if predictions[i] * labels[i] < 0:
                error += 1

        return error







