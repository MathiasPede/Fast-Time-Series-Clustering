# -*- coding: UTF-8 -*-
"""
ed_c
~~~~~~~~~~~~~~~~~~
Euclidean Distance (ED), C implementation.
Based on dtw_c from dtaidistance
:author: Mathias Pede
"""

import logging
import math
import numpy as np
cimport numpy as np
cimport cython
import cython
import ctypes
from cpython cimport array, bool
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free, abs, labs
from libc.stdio cimport printf
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, pow
from libc.stdint cimport intptr_t
from cpython.exc cimport PyErr_CheckSignals
from dtaidistance.util import SeriesContainer

logger = logging.getLogger("be.kuleuven.dtai.distance")

DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef double inf = np.inf

def ed_nogil(double[:] s1, double[:] s2):
    """Euclidean Distance (ED).
    This calls a pure c dtw computation that avoids the GIL.
    :param s1: First sequence (buffer of doubles)
    :param s2: Second sequence (buffer of doubles)
    """
    #return distance_nogil_c(s1, s2, len(s1), len(s2),
    # If the arrays (memoryviews) are not C contiguous, the pointer will not point to the correct array
    if isinstance(s1, (np.ndarray, np.generic)):
        if not s1.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s1 = s1.copy()
    if isinstance(s2, (np.ndarray, np.generic)):
        if not s2.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 2 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s2 = s2.copy()
    length = min(len(s1), len(s2))
    return ed_nogil_c(&s1[0], &s2[0], length)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef double ed_nogil_c(
                double *s1, double *s2,
                int l #length s
                ) nogil:
    """Euclidean distance.
    
    Pure c msm implementation that avoids GIL.
    """

    cdef double temp = 0
    cdef double diff
    cdef int i

    for i in range(l):
        diff = s2[i] - s1[i]
        temp = temp + diff*diff

    return sqrt(temp)

def distance_matrix_nogil(cur, block=None,
                          bool is_parallel=False):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL.
    :return: The distance matrix as a list representing the triangular matrix.
    """
    # https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
    # Prepare for only c datastructures
    cdef int length = 0
    cdef int block_rb=0
    cdef int block_re=0
    cdef int block_cb=0
    cdef int block_ce=0
    cdef ri = 0
    if block is not None and block != 0.0:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]
        for ri in range(block_rb, block_re):
            if block_cb <= ri:
                if block_ce > ri:
                    length += (block_ce - ri - 1)
            else:
                if block_ce > ri:
                    length += (block_ce - block_cb)
    else:
        length = int(len(cur) * (len(cur) - 1) / 2)
    cdef double large_value = inf

    dists_py = np.full((length,), large_value, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] dists = dists_py
    #print('dists: {}, {}'.format(dists_py.shape, dists_py.shape[0]*dists_py.shape[1]))
    cdef double **cur2 = <double **> malloc(len(cur) * sizeof(double*))
    cdef int *cur2_len = <int *> malloc(len(cur) * sizeof(int))
    # cdef long ptr;
    cdef intptr_t ptr
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] cur_np;

    if cur.__class__.__name__ == "SeriesContainer":
        cur = cur.c_data()
    if type(cur) in [list, set]:
        for i in range(len(cur)):
            ptr = cur[i].ctypes.data
            cur2[i] = <double *> ptr
            cur2_len[i] = len(cur[i])
    elif isinstance(cur, np.ndarray):
        if not cur.flags.c_contiguous:
            logger.debug("Warning: The numpy array or matrix passed to method distance_matrix is not C-contiguous. " +
                         "The array will be copied.")
            cur = cur.copy(order='C')
        cur_np = cur
        for i in range(len(cur)):
            cur2[i] = &cur_np[i,0]
            cur2_len[i] = cur_np.shape[1]
    else:
        return None

    if is_parallel:
        distance_matrix_nogil_c_p(cur2, len(cur), cur2_len, &dists[0],
                                  block_rb, block_re, block_cb, block_ce)
    else:
        distance_matrix_nogil_c(cur2, len(cur), cur2_len, &dists[0],
                                block_rb, block_re, block_cb, block_ce)
    free(cur2)
    free(cur2_len)
    return dists_py


cdef distance_matrix_nogil_c(double **cur, int len_cur, int* cur_len, double* output,
                             int block_rb=0, int block_re=0, int block_cb=0, int block_ce=0):
    cdef int r
    cdef int c
    cdef int cb
    cdef int i

    if block_re == 0:
        block_re = len_cur
    if block_ce == 0:
        block_ce = len_cur
    i = 0
    for r in range(block_rb, block_re):
        if r + 1 > block_cb:
            cb = r+1
        else:
            cb = block_cb
        for c in range(cb, block_ce):
            output[i] = ed_nogil_c(cur[r], cur[c], cur_len[r])
            i += 1


cdef distance_matrix_nogil_c_p(double **cur, int len_cur, int* cur_len, double* output,
                               int block_rb=0, int block_re=0, int block_cb=0, int block_ce=0):
    # Requires openmp which is not supported for clang on mac by default (use newer version of clang)
    cdef Py_ssize_t r
    cdef Py_ssize_t c
    cdef Py_ssize_t cb
    cdef Py_ssize_t brb = block_rb  # TODO: why is this necessary for cython?
    cdef Py_ssize_t ir
    cdef long llength = 0
    cdef Py_ssize_t slength

    if block_re == 0 and block_ce == 0:
        # First divide the even number to avoid overflowing
        if len_cur % 2 == 0:
            llength = (len_cur / 2) * (len_cur - 1)
        else:
            llength = len_cur * ((len_cur - 1) / 2)
    else:
        for ri in range(block_rb, block_re):
            if block_cb <= ri:
                if block_ce > ri:
                    llength += (block_ce - ri - 1)
            else:
                if block_ce > ri:
                    llength += (block_ce - block_cb)
    length = llength
    if length < 0:
        print("ERROR: Length of array needed to represent the distance matrix larger than maximal value for Py_ssize_t")
        return

    cdef double large_value = inf

    if block_re == 0:
        block_re = len_cur
    if block_ce == 0:
        block_ce = len_cur

    cdef np.ndarray[np.intp_t, ndim=1, mode="c"] irs_py = np.empty((length,), dtype=np.intp)
    cdef Py_ssize_t[:] irs = irs_py
    cdef np.ndarray[np.intp_t, ndim=1, mode="c"] ics_py = np.empty((length,), dtype=np.intp)
    cdef Py_ssize_t[:] ics = ics_py
    ir = 0
    for r in range(brb, block_re):
        if r + 1 > block_cb:
            cb = r+1
        else:
            cb = block_cb
        for c in range(cb, block_ce):
            irs[ir] = r
            ics[ir] = c
            ir += 1

    with nogil, parallel():
        for ir in prange(length):
            r = irs[ir]
            c = ics[ir]
            output[ir] = ed_nogil_c(cur[r], cur[c], cur_len[r])