__author__ = 'harnoor'

import numpy as np
import heapq
import operator


class KNN:
    train_data = None
    k = 0
    report = False

    # Takes the euclidean distance between x and y
    def euclideanDistance(x1, x2):
        distance = np.sum(np.subtract(x1, x2) ** 2)
        return distance

    # Classify sample
    def classifySample(self, x, distanceFunction=euclideanDistance):
        trainX = self.train_data[:, 1:]
        trainY = self.train_data[:, 0]

        difference = np.apply_along_axis(distanceFunction, 1, trainX, x)

        tupleList = list()
        for i in range(len(difference)):
            tupleList.append((difference[i], i))
            
        smallest = heapq.nsmallest(self.k, tupleList)

        neighbors = dict()
        for j in range(len(smallest)):
            index = smallest[j][1]
            if trainY[index] in neighbors:
                neighbors[trainY[index]] += 1
            else:
                neighbors[trainY[index]] = 1
        classification = max(neighbors.iteritems(), key=operator.itemgetter(1))[0]
        if self.report:
            print "Classified as %d" % classification

        return classification

    # Store the model in a dictionary
    # train_data is a matrix of where the first column holds the 
    # Y values and the other columns are features.
    def __init__(self, train_data, k=1, report=True):
        self.k = k
        self.train_data = train_data
        self.report = report

