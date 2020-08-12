import numpy as np
import logging

DTYPE = np.double
logger = logging.getLogger("timeseries.distances")


def distances_matrix_to_array_with_diagonal(matrix):
    size = len(matrix)
    flat = np.zeros((size + 1) * size // 2)

    for i in range(size):
        for j in range(i, size):
            index = (((2 * size - i + 1) * i) // 2) + (j - i)
            flat[index] = matrix[i, j]

    return flat


def distances_array_to_matrix_with_diagonal(array, size):
    matrix = np.full((size, size), np.inf, dtype=DTYPE)

    for i in range(size):
        for j in range(i, size):
            index = (((2 * size - i + 1) * i) // 2) + (j - i)
            matrix[i, j] = array[index]
            matrix[j, i] = array[index]

    return matrix


def distances_array_to_matrix(dists, nb_series, block=None):
    """Transform a condensed distances array to a full matrix representation.
    The upper triangular matrix will contain all the distances.
    """
    dists_matrix = np.full((nb_series, nb_series), np.inf, dtype=DTYPE)
    idxs = _distance_matrix_idxs(block, nb_series)
    dists_matrix[idxs] = dists
    # dists_cond = np.zeros(self._size_cond(len(series)))
    # idx = 0
    # for r in range(len(series) - 1):
    #     dists_cond[idx:idx + len(series) - r - 1] = dists[r, r + 1:]
    #     idx += len(series) - r - 1
    return dists_matrix


def _distance_matrix_idxs(block, nb_series):
    if block is None or block == 0:
        idxs = np.triu_indices(nb_series, k=1)
    else:
        idxsl_r = []
        idxsl_c = []
        for r in range(block[0][0], block[0][1]):
            for c in range(max(r + 1, block[1][0]), min(nb_series, block[1][1])):
                idxsl_r.append(r)
                idxsl_c.append(c)
        idxs = (np.array(idxsl_r), np.array(idxsl_c))
    return idxs


def _distance_matrix_length(block, nb_series):
    if block is not None:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]
        length = 0
        for ri in range(block_rb, block_re):
            if block_cb <= ri:
                if block_ce > ri:
                    length += (block_ce - ri - 1)
            else:
                if block_ce > ri:
                    length += (block_ce - block_cb)
    else:
        length = int(nb_series * (nb_series - 1) / 2)
    return length


def _print_library_missing(raise_exception=True):
    msg = "The compiled C library is not available.\n" +\
          "See the documentation for alternative installation options."
    logger.error(msg)
    if raise_exception:
        raise Exception(msg)