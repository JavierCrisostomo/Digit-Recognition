__author__ = 'harnoor'
import numpy as np
import heapq
import operator


class Classifier:
    train_data = None
    k = 0
    report = False
    weighted = False

    # Takes the euclidean distance between x and y
    def euclideanDistance(x1, x2):
        distance = np.sum(np.subtract(x1, x2) ** 2)
        return distance

    def kNearestNeighbors(self, x, k, distanceFunction=euclideanDistance):
        trainX = self.train_data[:, 1:]
        difference = np.apply_along_axis(distanceFunction, 1, trainX, x)
        tupleList = list()
        for i in range(len(difference)):
            tupleList.append((difference[i], i))
        smallest = heapq.nsmallest(k, tupleList)
        return smallest

    def classifySampleGivenNeighbors(self, x, smallest, distanceFunction=euclideanDistance):
        trainY = self.train_data[:, 0]
        trainX = self.train_data[:, 1:]
        neighbors = dict()
        for j in range(len(smallest)):
            index = smallest[j][1]
            if trainY[index] in neighbors:
                weight = 1
                if self.weighted:
                    weight = (1/distanceFunction(x, trainX[index]))
                neighbors[trainY[index]] += weight*1
            else:
                neighbors[trainY[index]] = 1
        classification = max(neighbors.iteritems(), key=operator.itemgetter(1))[0]
        return classification


    # Classify sample
    def classifySample(self, x, distanceFunction=euclideanDistance):
        trainX = self.train_data[:, 1:]
        trainY = self.train_data[:, 0]

        smallest = self.kNearestNeighbors(x, self.k, distanceFunction)
        classification = self.classifySampleGivenNeighbors(x, smallest)


        if self.report:
            print "Classified as %d" % classification

        return classification

    # Store the model in a dictionary
    # train_data is a matrix of where the first column holds the 
    # Y values and the other columns are features.
    def __init__(self, train_data, k=1, report=True, weighted=False):
        self.k = k
        self.train_data = train_data
        self.report = report
        self.weighted = weighted

