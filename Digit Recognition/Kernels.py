import numpy as np
import math


def defaultKernel(x_m, x_t, dim):
    product = np.dot(x_m, np.transpose(x_t))
    return product


def exactPolyKernel(x_m, x_t, dim):
    product = np.dot(x_m, x_t) ** dim
    return product


def polyKernel(x_m, x_t, dim):
    product = (np.dot(x_m, np.transpose(x_t)) + 1) ** dim
    return product


def exponentialKernel(x_m, x_t, sigma):
    numerator = math.sqrt(np.dot((x_m - x_t), (x_m - x_t)))
    denominator = 2 * sigma ** 2
    return math.exp(-1 * (numerator / denominator))
