__author__ = 'harnoor'

import numpy as np
import heapq
import operator

class KNN:
    train_data = None
    k = 0
    report = False

    # Takes the euclidean distance between x and y
    def euclideanDistance(x, y):
        if type(x) is not None:
            print "ERROR: %s" % type(x)
        if type(y) is not None:
            print "ERROR Y: %s" % type(y)
        return (x - y)**2

    # Classify sample
    def classifySample(self, x, distanceFunction=euclideanDistance):
        trainX = self.train_data[:, 1:]
        trainY = self.train_data[:,0]
        
        print "mapping"
        difference = map(distanceFunction, x, trainX)
        print "mapped"

        tupleList = list()
        for i in range(len(difference)):
            tupleList.append(difference[i], i)            
            
        smallest = heapq.nsmallest(self.k, tupleList)

        neighbors = dict()
        for j in len(smallest):
            if trainY[j] in x:
                neighbors[trainY[j]] += 1
            else:
                neighbors[trainY[j]] = 1
        classification = max(neighbors.iteritems(), key=operator.itemgetter(1))[0]
        if self.report:
            print "Classified as %d" % classification
        return classification


    # Store the model in a dictionary
    def initializeKNN(self, train_data, k=1, report=True):
        self.k = k
        self.train_data = train_data
        self.report = report

    def __init__(self):
        pass
