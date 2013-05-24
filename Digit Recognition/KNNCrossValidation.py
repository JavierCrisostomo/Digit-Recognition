import numpy as np
import KNN
import sys

# Use cross validation to fine tune k
# k_values is a list of values of k to consider
def crossValidate(k_values, train_data):
    train = train_data
    errors = [0] * len(k_values)

    # For each k we want to cross validate
    for i in range(len(k_values)):
        k = k_values[i]

        # Exclude k samples from our training set
        for j in range(len(train_data)/k):
            index = j * k
            to_test = train[index:(index + k),:]
            to_train = np.delete(train, range(index, index + k), axis=0)
            knn = KNN.KNN(to_train, k)
            
            for x in to_test:
                x_sample = x[1:]
                y = x[0]
                if knn.classifySample(x_sample) != y:
                    errors[i] += 1
    print errors
    return errors.index(min(errors))

                
            
        

