import numpy as np
import KNN
import multiprocessing as mp


# Helper method used by each process to perform one full cross validation sweep
# across the training data.  In other words, it will rotate across all of the
# folds and test with each one while averaging the error.  It will then return the
# error along with the k value.  It accepts the training data, the number of folds
# to use and the value of k to use for this classification.
def _classifyData(train, num_folds, k):
    # determine size of folds
    fold_size = len(train) / num_folds

    error = 0

    # split the data into training and testing
    for j in range(num_folds):
        index = j * fold_size

        # pull out testing data
        to_test = train[index:(index + fold_size), :]

        # remove testing data from training data
        to_train = np.delete(train, range(index, index + fold_size), axis=0)

        # create classifier with training data and k value
        knn = KNN.Classifier(to_train, k, False)

        # classify each test sample and calculate average error
        for x in to_test:
            x_sample = x[1:]
            y = x[0]
            if knn.classifySample(x_sample) != y:
                error += 1.0 / (fold_size * (num_folds - 1))

    return error, k

# Performs K-Fold Cross Validation for KNN.  It will detect the number of available
# CPU cores and run the validation in parallel.  It will assign a different data set
# to each core and a different value of k.  It will then compare all the resulting
# errors and return the value of k resulting in the smallest error.  It accepts a list
# of k values with which to use for validation, the number of folds to split the data
# into testing and training sets, the training data itself and the number of cores to
# multithread across.  It will automatically detect the number of cores if a number is
# not passed.
def KNNCrossValidation(k_values, num_folds, train_data, cores=mp.cpu_count()):
    train = train_data
    errors = [0] * len(k_values)

    # Create process pool
    pool = mp.Pool(cores)
    workers = list()

    # For each k we want to cross validate
    for i in range(len(k_values)):
        k = k_values[i]

        # run the data classification for this data set and k value on a separate
        # process
        p = pool.apply_async(_classifyData, args=(train, num_folds, k))
        workers.append(p)

    # close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # obtain all the results for each process
    for w in workers:
        error, index = w.get()
        errors[k_values.index(index)] = error

    # return k value with smallest error
    return k_values[errors.index(min(errors))]

                
            
        

