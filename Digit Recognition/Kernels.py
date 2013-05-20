import numpy as np
import math


def defaultKernel():
    def applyKernel(x_m, x_t):
        product = np.dot(x_m, x_t)
        return product
    return applyKernel


def exactPolyKernel(dim):
    def applyKernel(x_m, x_t):
        product = np.dot(x_m, x_t) ** dim
        return product
    return applyKernel


def polyKernel(dim):
    def applyKernel(x_m, x_t):
        product = (np.dot(x_m, np.transpose(x_t)) + 1) ** dim
        return product
    return applyKernel


def exponentialKernel(sigma):
    def applyKernel(x_m, x_t):
        numerator = math.sqrt(np.dot((x_m - x_t), (x_m - x_t)))
        denominator = 2 * sigma ** 2
        return math.exp(-1 * (numerator / denominator))
    return applyKernel
