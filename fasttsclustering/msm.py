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
from helper import _print_library_missing, distances_array_to_matrix, _distance_matrix_length

from dtaidistance.util import SeriesContainer, dtaidistance_dir

logger = logging.getLogger("timeseries.distance")

msm_c = None
try:
    import msm_c
except ImportError:
    # logger.info('C library not available')
    msm_c = None

DTYPE = np.double


def msm_fast(s1, s2, penalty=None):
    """Fast C version of :meth:`distance`.
    Note: the series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or
    ``array.array('d', [1,2,3])``
    """
    if msm_c is None:
        _print_library_missing()
        return None
    if penalty is None:
        penalty = 0.1
    d = msm_c.msm_nogil(s1, s2, penalty=penalty)
    return d


def msm_matrix_fast(s, penalty=0.1,
                         block=None, parallel=True):
    """Fast C version of :meth:`distance_matrix`."""
    if msm_c is None:
        _print_library_missing()
        return None

    s = SeriesContainer.wrap(s)

    logger.info("Compute distances in pure C (parallel={})".format(parallel))
    dists = msm_c.distance_matrix_nogil(s, penalty=penalty, block=block, is_parallel=parallel)

    exp_length = _distance_matrix_length(block, len(s))
    assert len(dists) == exp_length, "len(dists)={} != {}".format(len(dists), exp_length)

    # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
    dists_matrix = distances_array_to_matrix(dists, nb_series=len(s), block=block)

    return dists_matrix


def try_import_c():
    global msm_c
    try:
        import msm_c
    except ImportError as exc:
        print('Cannot import C library')
        print(exc)
        msm_c = None


if __name__ == '__main__':
    import distance_functions as df
    t1 = np.arange(1,1000, dtype=np.double)
    t2 = np.arange(1000,1,-1, dtype=np.double)

    result = df.msm(t1,t2,c=0.1)
    print(result)

    result_fast = msm_fast(t1,t2,penalty=0.1)
    print(result_fast)

    result_dtw = df.dtw(t1,t2)
    print(result_dtw)

