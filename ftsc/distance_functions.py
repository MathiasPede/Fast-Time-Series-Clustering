"""
File containing the the distance/similarity functions to compare different time series
# Implemented functions:
#   Euclidean Distance (ED)
#   Dynamic Time Warping (DTW)
#   Move Split Merge (MSM)
"""

from dtaidistance.dtw import distance_fast, distance_matrix_fast
from .ed import ed, ed_matrix
from .msm import msm, msm_matrix


def dtw(s1, s2, args: dict):
    if 'window' in args:
        window = args['window']
    else:
        window=None
    return distance_fast(s1, s2, window=window)


def dtw_matrix(series, args: dict):
    if 'window' in args:
        window = args['window']
    else:
        window=None
    if 'block' in args:
        block = args['block']
    else:
        block=None
    return distance_matrix_fast(series, window=window, block=block, )


functions = {
    'ed': (ed, ed_matrix),
    'msm': (msm, msm_matrix),
    'dtw': (dtw, dtw_matrix)
}


def compute_distance(s1, s2, func, args=None):
    if args is None:
        args={}
    compare = functions[func][0]
    return compare(s1, s2, args)


def compute_distance_matrix(series, func, args=None):
    if args is None:
        args={}
    compare = functions[func][1]
    return compare(series, args)