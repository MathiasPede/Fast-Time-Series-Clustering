"""
File containing the the distance/similarity functions to compare different time series
# Implemented functions:
#   Euclidean Distance (ED)
#   Dynamic Time Warping (DTW)
#   Move Split Merge (MSM)
"""

from dtaidistance import dtw as dtai_dtw
import numpy as np
from math import sqrt


def ed(s1, s2):
    """
    Implementation of euclidean distance for time series.
    Raises an exception when both time series aren't of equal length
    :param s1: First time series
    :param s2: Second time series
    :return: ED Distance
    """
    # Check equal length time series
    l1, l2 = len(s1), len(s2)
    if l1 != l2:
        print("Time series have unequal length: {} and {}".format(l1, l2))
        raise

    diff_array = np.subtract(s2, s1)
    squared_array = np.square(diff_array)
    sum_of_differences = np.sum(squared_array)

    return sqrt(sum_of_differences)


def dtw(s1, s2, window=None):
    """
    Implementation of DTW distance for time series
    Simply uses the fast C implementation by dtaidistance
    :param s1: First time series
    :param s2: Second time series
    :param window: Warp window used by the algorithm
    :return: DTW Distance
    """
    return dtai_dtw.distance_fast(s1, s2, window=window)


def msm(s1, s2, c=0.1):
    """
    Implementation of Move Split Merge for time series
    :param s1: First time series
    :param s2: Second time series
    :param c: Cost penalty for executing the Split or Merge operation
    :return: MSM Distance
    """

    l1, l2 = len(s1), len(s2)
    Cost = np.zeros(shape=(l1, l2))  # Empty matrix

    Cost[0, 0] = abs(s1[0] - s2[0])

    # Initialize the first row
    for i in range(1, l1):
        Cost[i, 0] = Cost[i - 1, 0] + _msm_f(s1[i], s1[i - 1], s2[1], c)

    # Initialize the first column
    for j in range(1, l2):
        Cost[0, j] = Cost[0, j - 1] + _msm_f(s2[j], s1[1], s2[j - 1], c)

    # Compute rest of matrix
    for i in range(1, l1):
        for j in range(1, l2):
            Cost[i, j] = min(Cost[i - 1, j - 1] + abs(s1[i] - s2[j]),
                             Cost[i - 1, j] + _msm_f(s1[i], s1[i - 1], s2[j], c),
                             Cost[i, j - 1] + _msm_f(s2[j], s1[i], s2[j - 1], c))

    # The cost in the bottom right corner is the MSM distance
    return Cost[l1 - 1, l2 - 1]


def _msm_f(x, y, z, cost):
    """
    Helper function for MSM distance calculation
    """
    if (y <= x <= z) or (y >= x >= z):
        return cost
    else:
        return cost + min(abs(x - y), abs(x - z))


# Convert the distance function to a similarity function
def calculate_dtw_similarity(serie1, serie2, window=None):
    term1 = dtai_dtw.distance_fast(serie1, np.zeros(1))
    term2 = dtai_dtw.distance_fast(serie2, np.zeros(1))
    term3 = dtai_dtw.distance_fast(serie1, serie2, window=window)

    return (term1*term1 + term2*term2 - term3*term3) / (2.0 * term1 * term2)

# a = np.array([1, 2, 3, 4, 5, 6], dtype=float)
# b = np.array([1, 2, 3, 5, 6], dtype=float)
#
# print(dtw(a, b))
# print(msm(a, b, 2))
