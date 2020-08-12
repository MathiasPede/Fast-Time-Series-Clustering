import logging
import math
import numpy as np
cimport numpy as np
cimport cython
import cython
import ctypes
from cpython cimport array, bool
from libc.stdlib cimport abort, malloc, free, abs, labs
from libc.stdio cimport printf
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, pow
from libc.stdint cimport intptr_t
from cpython.exc cimport PyErr_CheckSignals

logger = logging.getLogger("timeseries.trianglefixing")

DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef double inf = np.inf

def triangle_fixing_nogil(double[:] matrix, int size, double violation_margin=0.01):
    if isinstance(matrix, (np.ndarray, np.generic)):
        if not matrix.base.flags.c_contiguous:
            logger.debug("Warning: Matrix passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            matrix = matrix.copy()

    length = (size + 1) * size // 2
    cdef double large_value = inf
    output_py = np.full((length,), large_value, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] output = output_py

    triangle_fixing_nogil_c(&matrix[0], size, &output[0], violation_margin)

    return output_py

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef int triangle_fixing_nogil_c(double *matrix, int size, double * output,
                             double violation_margin=0.01) nogil:
    cdef double * errors
    cdef double * correction_terms
    cdef int flat_matrix_size = ((size+1) * size) / 2

    # Initialize error and correction matrix
    errors = <double *> malloc(sizeof(double) * flat_matrix_size)
    correction_terms = <double *> malloc(sizeof(double) * flat_matrix_size * size)

    cdef int i
    cdef int j
    cdef int k

    for i in range(0, flat_matrix_size):
        errors[i] = 0.0
    for i in range(0, flat_matrix_size * size):
        correction_terms[i] = 0.0

    #printf("Initialization done\n")

    # Loop parameter
    cdef int violations = size

    cdef int ki_index
    cdef int jk_index
    cdef int ij_index
    cdef double b
    cdef double ki_error
    cdef double jk_error
    cdef double ij_error
    cdef double mu
    cdef double zijk
    cdef double theta

    while violations > 0:
        violations = 0
        for i in range(0,size):
            for j in range(i+1, size):

                ij_index = get_flat_index_nogil_c(i, j, size)

                for k in range(0,size):
                    if i==k or j==k:
                        continue

                    ki_index = get_flat_index_nogil_c(k, i, size)
                    jk_index = get_flat_index_nogil_c(j, k, size)


                    b = matrix[ki_index] + matrix[jk_index] - matrix[ij_index]

                    ki_error = errors[ki_index]
                    jk_error = errors[jk_index]
                    ij_error = errors[ij_index]

                    mu = (1.0 / 3.0) * (ij_error - jk_error - ki_error - b)

                    if mu > 0:
                        if mu > violation_margin:
                            violations = violations + 1
                        zijk = correction_terms[ij_index*size + k]

                        theta = -mu
                        if zijk < theta:
                            theta = zijk

                        errors[ij_index] = ij_error + theta
                        errors[jk_index] = jk_error - theta
                        errors[ki_index] = ki_error - theta

                        correction_terms[ij_index*size + k] = zijk - theta

        #printf("Number of triangle violations: %d\n", violations)

    #printf("Computing fixed matrix\n")
    for i in range(0, flat_matrix_size):
        output[i] = matrix[i] + errors[i]

    free(errors)
    free(correction_terms)
    return 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef int get_flat_index_nogil_c(int i, int j, int size) nogil:
    cdef int first = i
    cdef int second = j
    if i > j:
        first = j
        second = i
    return (((2*size - first + 1) * first) / 2) + (second-first)


def get_number_of_violations_nogil(double[:] matrix, int size, double violation_margin=0.01):
    if isinstance(matrix, (np.ndarray, np.generic)):
        if not matrix.base.flags.c_contiguous:
            logger.debug("Warning: Matrix passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            matrix = matrix.copy()

    result = get_number_of_violations_nogil_c(&matrix[0], size, violation_margin)

    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef int get_number_of_violations_nogil_c(double *matrix, int size,
                             double violation_margin=0.01) nogil:

    cdef int i
    cdef int j
    cdef int k

    # Loop parameter
    cdef int violations = 0

    cdef int ki_index
    cdef int jk_index
    cdef int ij_index
    cdef double b

    for i in range(0,size):
        for j in range(i+1, size):

            ij_index = get_flat_index_nogil_c(i, j, size)

            for k in range(0,size):
                if i==k or j==k:
                    continue

                ki_index = get_flat_index_nogil_c(k, i, size)
                jk_index = get_flat_index_nogil_c(j, k, size)

                b = matrix[ki_index] + matrix[jk_index] - matrix[ij_index]

                if -b > violation_margin:
                    violations = violations + 1

    #printf("Number of triangle violations: %d\n", violations)

    return violations