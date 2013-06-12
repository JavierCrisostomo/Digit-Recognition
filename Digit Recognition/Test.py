__author__ = 'bdwalker'

import KNN
import Perceptron
import numpy as np


class TestModel:

    # Passed an unpacked model_dict
    def predictWithModel(self, model_dict, test_data, kernel, report=False):
        perceptron = Perceptron.Classifier()
        
        votes = np.zeros(shape=(test_data.shape[0], 10), dtype = int)

        for couple in model_dict:
            predictions = perceptron.testWithModel(test_data, kernel, model_dict[couple])
            print predictions
            for i in range(predictions.shape[0]):
                if predictions[i] == 1:
                    votes[i, couple[0]] += 1
                else:
                    votes[i, couple[1]] += 1

        classifications = list()

        for i in range(len(votes)):
            classification = str(votes[i].argmax()) + "\n"
            print "Classified ", classification
            classifications.append(classification)
        return classifications

    def calculateLoss(self, predictions, labels):
        error = 0
        for i in range(0, len(predictions)):
            if predictions[i] * labels[i] < 0:
                error += 1
        return error







