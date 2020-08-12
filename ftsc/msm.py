# -*- coding: UTF-8 -*-
"""
msm
~~~~~~~~~~~~~~~~
Move Split Merge (MSM)
Based on dtw from dtaidistance
:author: Mathias Pede
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
import math
import numpy as np
from .util import _print_library_missing, distances_array_to_matrix, _distance_matrix_length

from dtaidistance.util import SeriesContainer, dtaidistance_dir

logger = logging.getLogger("ftsc")

msm_c = None
try:
    import msm_c
except ImportError:
    # logger.info('C library not available')
    msm_c = None

DTYPE = np.double


def msm(s1, s2, args: dict):
    """
    Move Split Merge

    Uses C version of MSM to compute the metric distance between two Time Series
    If C library can't be accessed, python version is used instead
    @param s1: First Time Series
    @param s2: Second Time Series
    @param args: dictionary, optionally containing penalty parameter
    @return: MSM distance between s1 and s2
    """
    if 'c' in args:
        penalty = args['c']
    else:
        penalty = 0.1

    if msm_c is None:
        _print_library_missing()
        logger.error("C library missing, using Python instead")
        return msm_py(s1, s2, c=penalty)

    return msm_fast(s1, s2, penalty=penalty)


def msm_fast(s1, s2, penalty=None):
    """Fast C version of MSM.
    Note: Time Series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or
    ``array.array('d', [1,2,3])``
    """
    if penalty is None:
        penalty = 0.1
    d = msm_c.msm_nogil(s1, s2, penalty=penalty)
    return d


def msm_py(s1, s2, c=0.1):
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


def msm_matrix(series, args, block=None, parallel=True):
    if 'c' in args:
        penalty = args['c']
    else:
        penalty = 0.1

    if msm_c is None:
        _print_library_missing()
        logger.error("C library missing, using Python instead")
        return msm_matrix_py(series, penalty=penalty)

    return msm_matrix_fast(series, penalty=penalty, block=block, parallel=parallel)


def msm_matrix_fast(s, penalty=0.1, block=None, parallel=True):
    """Fast C version of MSM distance_matrix`."""
    if msm_c is None:
        _print_library_missing()
        return None

    if penalty is None:
        penalty=0.1

    s = SeriesContainer.wrap(s)

    logger.info("Compute distances in pure C (parallel={})".format(parallel))
    dists = msm_c.distance_matrix_nogil(s, penalty=penalty, block=block, is_parallel=parallel)

    exp_length = _distance_matrix_length(block, len(s))
    assert len(dists) == exp_length, "len(dists)={} != {}".format(len(dists), exp_length)

    # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
    dists_matrix = distances_array_to_matrix(dists, nb_series=len(s), block=block)

    return dists_matrix


def msm_matrix_py(series, penalty=0.01):
    size = len(series)
    result = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(i, size):
            result[i, j] = msm(series[i], series[j], {'c': penalty})
            result[j, i] = result[i, j]
    return result


def try_import_c():
    global msm_c
    try:
        import msm_c
    except ImportError as exc:
        print('Cannot import C library')
        print(exc)
        msm_c = None