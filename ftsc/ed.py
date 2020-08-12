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
import math
import numpy as np
from helper import distances_array_to_matrix, _distance_matrix_length, _print_library_missing

from dtaidistance.util import SeriesContainer, dtaidistance_dir

logger = logging.getLogger("timeseries.distance")

ed_c = None
try:
    import ed_c
except ImportError:
    # logger.info('C library not available')
    ed_c = None

DTYPE = np.double


def ed_fast(s1, s2):
    """Fast C version of :meth:`distance`.
    Note: the series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or
    ``array.array('d', [1,2,3])``
    """
    if ed_c is None:
        _print_library_missing()
        return None
    d = ed_c.ed_nogil(s1, s2)
    return d


def ed_matrix_fast(s, block=None, parallel=True):
    """Fast C version of :meth:`distance_matrix`."""
    if ed_c is None:
        _print_library_missing()
        return None

    s = SeriesContainer.wrap(s)

    logger.info("Compute distances in pure C (parallel={})".format(parallel))
    dists = ed_c.distance_matrix_nogil(s, block=block, is_parallel=parallel)

    exp_length = _distance_matrix_length(block, len(s))
    assert len(dists) == exp_length, "len(dists)={} != {}".format(len(dists), exp_length)

    # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
    dists_matrix = distances_array_to_matrix(dists, nb_series=len(s), block=block)

    return dists_matrix


def try_import_c():
    global ed_c
    try:
        import ed_c
    except ImportError as exc:
        print('Cannot import C library')
        print(exc)
        ed_c = None


if __name__ == '__main__':
    import distance_functions as df
    t1 = np.arange(1,1000, dtype=np.double)
    t2 = np.arange(1000,1,-1, dtype=np.double)

    result = df.msm(t1,t2,c=0.1)
    print(result)

    result_fast = ed_fast(t1,t2)
    print(result_fast)

    result_dtw = df.dtw(t1,t2)
    print(result_dtw)

