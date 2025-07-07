import numpy as np


def get_squared_distance_matrix(x, diag_value=0.0):
    """
    Return the matrix of squared euclidean distances

    :param x: array representing a set of N points in a vector space
    :param diag_value: value to put on the diagonal of the matrix
    :return: NxN matrix containing the squared euclidean distance for each pair of points
    """
    x1 = np.expand_dims(x, 1)
    x0 = np.expand_dims(x, 0)
    d2 = np.sum((x1 - x0) ** 2, 2)
    d2[np.diag_indices_from(d2)] = diag_value
    return d2
