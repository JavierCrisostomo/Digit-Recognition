__author__ = 'bdwalker'

import KNN
import Perceptron
import numpy as np


class TestModel:

    def _tallyVotes(self, predictions):
        lables = [9] * predictions.shape[0]
        for i in range(predictions.shape[0]):
            lables[ which.predictions[i,:]

    def predictWithModel(self, model_dict, test_data, report=False):
        perceptron = Perceptron.Classifier()
        votes = np.ndarray(shape=(test_data.shape[0], 10), dtype = int)
        for couple in model_dict:
            predictions = perceptron.testWithModel(test_data, model_dict[couple])
            for i in range(predictions.shape[0]):
                if predictions[i] == 1:
                    votes[i, couple[0]] += 1
                else:
                    votes[i, couple[1]] += 1


        return predictions

    def calculateLoss(self, predictions, labels):
        error = 0
        for i in range(0, len(predictions)):
            if predictions[i] * labels[i] < 0:
                error += 1

        return error







