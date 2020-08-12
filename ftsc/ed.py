# -*- coding: UTF-8 -*-
"""
ed
~~~~~~~~~~~~~~~~
Euclidean Distance (ED)
Based on dtw from dtaidistance
:author: Mathias Pede
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
import numpy as np

from math import sqrt
from .util import distances_array_to_matrix, _distance_matrix_length, _print_library_missing

from dtaidistance.util import SeriesContainer, dtaidistance_dir

logger = logging.getLogger("timeseries.distance")

ed_c = None
try:
    import ftsc.ed_c as ed_c
except ImportError:
    # logger.info('C library not available')
    ed_c = None

DTYPE = np.double


def ed(s1, s2, args: dict):
    """
    Euclidean Distance

    Uses C version of ED to compute the metric distance between two Time Series
    If C library can't be accessed, python version is used instead
    @param s1: First Time Series
    @param s2: Second Time Series
    @param args: not relevant
    @return: ED distance between s1 and s2
    """
    if ed_c is None:
        _print_library_missing()
        logger.error("C library missing, using Python instead")
        return ed_py(s1, s2)

    if 'block' in args:
        block = args['block']
    else:
        block= None
    if 'compact' in args:
        compact = args['compact']
    else:
        compact = False

    return ed_fast(s1, s2)


def ed_fast(s1, s2):
    """Fast C version of ED`.
    Note: the series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or
    ``array.array('d', [1,2,3])``
    """
    if ed_c is None:
        _print_library_missing()
        return None
    d = ed_c.ed_nogil(s1, s2)
    return d


def ed_py(s1, s2):
    """
    Implementation of euclidean distance for time series.
    Raises an exception when both time series aren't of equal length
    :param s1: First time series
    :param s2: Second time series
    :return: ED Distance
    """
    diff_array = np.subtract(s2, s1)
    squared_array = np.square(diff_array)
    sum_of_differences = np.sum(squared_array)

    return sqrt(sum_of_differences)


def ed_matrix(series, args, block=None, parallel=True):
    if ed_c is None:
        _print_library_missing()
        logger.error("C library missing, using Python instead")
        return ed_matrix_py(series)

    if 'block' in args:
        block = args['block']
    else:
        block= None
    if 'compact' in args:
        compact = args['compact']
    else:
        compact = False

    return ed_matrix_fast(series, block=block, compact=compact, parallel=parallel)


def ed_matrix_fast(s, block=None, parallel=True, compact=False):
    """Fast C version of :meth:`distance_matrix`."""
    if ed_c is None:
        _print_library_missing()
        return None

    s = SeriesContainer.wrap(s)

    logger.info("Compute distances in pure C (parallel={})".format(parallel))
    dists = ed_c.distance_matrix_nogil(s, block=block, is_parallel=parallel)

    exp_length = _distance_matrix_length(block, len(s))
    assert len(dists) == exp_length, "len(dists)={} != {}".format(len(dists), exp_length)

    if compact:
        return dists

    # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
    dists_matrix = distances_array_to_matrix(dists, nb_series=len(s), block=block)

    return dists_matrix


def ed_matrix_py(series, penalty=0.01, block=None, compact=False):
    size = len(series)
    result = np.zeros((size, size), dtype=float)

    if block is None:
        for i in range(size):
            for j in range(i, size):
                result[i, j] = ed(series[i], series[j], {'c': penalty})
                result[j, i] = result[i, j]
    else:
        for i in range(block[0][0], block[0][1]):
            for j in range(block[1][0], block[1][0]):
                result[i, j] = ed(series[i], series[j], {'c': penalty})
                result[j, i] = result[i, j]

    if compact:
        return result.flatten()

    return result



def try_import_c():
    global ed_c
    try:
        import ftsc.ed_c as ed_c
    except ImportError as exc:
        print('Cannot import C library')
        print(exc)
        ed_c = None

