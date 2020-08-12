import numpy as np
import random as rnd
import logging

from math import sqrt, floor
from ftsc.cluster_problem import ClusterProblem

logger = logging.getLogger("ftsc")


def aca_symm(cp: ClusterProblem, tolerance=0.05, max_rank=None, start_index=None, seed=None, zero_diagonal=True):
    """
    Adaptive Cross Approximation for Symmetric Distance Matrices

    @param cp: Cluster Problem, includes the objects and a compare function
    @param tolerance: An estimated relative error of the resulting approximation
    @param max_rank: The maximal rank of the approximation (number of crosses)
    @param start_index: Optional first row to start the first cross
    @param seed: Optional seed to make algorithm deterministic

    @return: Approximation of the distance matrix of the cluster problem with an approximated relative error equal
    to the tolerance parameter.
    """
    if not 0.0 < tolerance < 1.0:
        logger.error("Opted tolerance not within [0.0,1.0] range")
    if not max_rank or max_rank > cp.cp_size():
        logger.debug("Max rank set to maximum")
        max_rank = cp.cp_size()
    if seed:
        rnd.seed(seed)

    # Calculate the rows and deltas of the crosses
    rows, deltas, m, error = aca_symmetric_body(cp, max_rank=max_rank, tolerance=tolerance,
                                                start_index=start_index)
    logger.debug("Found approximation of rank " + str(m) + "with estimated relative error of " + str(error))

    # Compute the full approximation
    matrix = calc_symmetric_matrix_approx(rows, deltas, m)

    # Fill the diagonal with zeros
    if zero_diagonal:
        np.fill_diagonal(matrix, 0)

    return matrix


def aca_symmetric_body(cp, max_rank=None, tolerance=0.05, start_index=None, iters_no_improvement=100):
    rows = []
    deltas = []

    sample_indices, sample_values = generate_samples_student_distribution(cp)
    sample_size = len(sample_values)

    # Threshold value is tolerance times the smallest sampled value
    initial_average = np.average(np.square(sample_values))
    best_remaining_average = initial_average
    max_allowed_relative_error = sqrt(initial_average) * tolerance

    # Start random row
    if start_index and 0 <= start_index <= cp.cp_size():
        logger.debug("Given start index " + str(start_index))
        pivot_index = start_index
    else:
        pivot_index = rnd.randint(0, cp.cp_size() - 1)

    # Loop variables
    indices = []
    m = 0
    best_m = 0
    stopcrit = False
    max_residu = 0.0
    deleted_indices = np.array([], dtype=int)
    restartable_samples = sample_values
    restartable_indices = sample_indices

    while m < max_rank and not stopcrit:
        logger.debug("Iteration " + str(m) + " New pivot: " + str(pivot_index))
        logger.debug("The indices so far: " + str(indices))

        # Calculate the current approximation for row of pivot
        approx = np.zeros(cp.cp_size())
        for i in range(m):
            approx = np.add(approx, rows[i] * rows[i][pivot_index] * (1.0 / deltas[i]))

        # Find new w vector
        new_row = np.subtract(cp.sample_row(pivot_index), approx)

        # Find delta at the pivot index
        new_delta = new_row[pivot_index]

        # If delta is zero, substitute by max value in the w vector
        if new_delta == 0:
            new_max = np.max(np.abs(new_row))
            # If the maximum is also 0 (row is perfectly approximated) take a new pivot from the samples
            if new_max == 0.0:
                index_sample_max = np.where(np.abs(restartable_samples) == max_residu)[0][0]
                pivot_index = restartable_indices[index_sample_max][0]
                continue
            new_delta = new_max

        # Add the cross
        indices.append(pivot_index)
        rows.append(new_row)
        deltas.append(new_delta)

        # Reevaluate the samples
        for j in range(sample_size):
            x = sample_indices[j, 0]
            y = sample_indices[j, 1]
            sample_values[j] = sample_values[j] - (1.0 / deltas[m]) * rows[m][y] * rows[m][x]

        # Estimate the frobenius norm and check stop criterion
        remaining_average = np.average(np.square(sample_values))
        stopcrit = (sqrt(remaining_average) < max_allowed_relative_error)

        # If average entry is lower the previous best, continue, otherwise check whether no improvement for x iters
        if remaining_average < best_remaining_average:
            best_remaining_average = remaining_average
            best_m = m
        elif m > best_m + iters_no_improvement:
            logger.debug("No improvement for 100 ranks, current rank: " + str(m))
            estimated_error = sqrt(best_remaining_average) / sqrt(initial_average)
            return rows, deltas, best_m, estimated_error

        # Delete the samples on the pivot row from the restartable samples
        pivot_indices_in_row_of_samples = np.where(sample_indices[:, 0] == pivot_index)[0]
        pivot_indices_in_col_of_samples = np.where(sample_indices[:, 1] == pivot_index)[0]
        pivot_indices_in_samples = np.concatenate((pivot_indices_in_row_of_samples, pivot_indices_in_col_of_samples))
        if deleted_indices.size == 0:
            deleted_indices = pivot_indices_in_samples
        else:
            deleted_indices = np.concatenate((deleted_indices, pivot_indices_in_samples), axis=0)
        restartable_samples = np.delete(sample_values, deleted_indices, axis=0)
        restartable_indices = np.delete(sample_indices, deleted_indices, axis=0)

        # Find the maximum error on the samples
        if restartable_samples.size == 0:
            max_residu = 0
        else:
            max_residu = np.max(np.abs(restartable_samples))

        # Choose a new pivot
        new_row_abs = np.abs(new_row)
        row_without_already_sampled_indices = np.delete(new_row_abs, indices, axis=0)
        new_max = np.max(row_without_already_sampled_indices)
        pivot_index = np.where(new_row_abs == new_max)[0][0]

        # Check whether the max of the row is smaller than the max residu from the samples, if so, switch
        if abs(new_max) < max_residu - 0.001:
            # Switch to the pivot to the sample with the largest remaining value
            index_sample_max = np.where(np.abs(restartable_samples) == max_residu)[0][0]
            pivot_index = restartable_indices[index_sample_max][0]

        m += 1

    estimated_error = sqrt(best_remaining_average) / sqrt(initial_average)

    if stopcrit:
        logger.debug("stopcrit: Approximated error: " + str(estimated_error))
    else:
        logger.debug("Max rank " + str(max_rank) + "achieved, Approximated error: " + str(estimated_error))

    return rows, deltas, best_m, estimated_error


def calc_symmetric_matrix_approx(rows, deltas, iters):
    rows_array = np.array(rows)[0:iters]
    deltas_array = np.array(deltas)[0:iters]
    cols = np.transpose(rows_array).copy()
    cols = np.divide(cols, deltas_array)
    result = np.matmul(cols, rows_array)
    return result


def generate_samples_student_distribution(cp, error_margin=0.01):
    amount_sampled = 0
    t = 3.39
    tolerance = np.infty
    size = cp.cp_size()

    sample_indices = None
    sample_values = None
    while tolerance > error_margin or amount_sampled < 2 * cp.cp_size():
        iteration_indices = np.zeros(shape=(size, 2), dtype=int)
        iteration_values = np.zeros(size, dtype=float)

        # Take size more samples
        for i in range(size):
            x = i
            y = i
            while x == y:
                y = rnd.randint(0, size - 1)
            iteration_indices[i, 0] = x
            iteration_indices[i, 1] = y
            iteration_values[i] = cp.sample(x, y)

        # Add the samples to the already sampled values
        if amount_sampled == 0:
            sample_indices = iteration_indices
            sample_values = iteration_values
        else:
            sample_indices = np.concatenate((sample_indices, iteration_indices))
            sample_values = np.concatenate((sample_values, iteration_values))

        # If sample size becomes too large, stop
        amount_sampled += size
        if amount_sampled > cp.cp_size() * cp.cp_size():
            break

        # Calculate the new current error margin
        squared = np.square(sample_values)
        average_so_far = np.mean(squared)
        std_so_far = np.std(squared)
        tolerance = (t * std_so_far) / (sqrt(amount_sampled) * average_so_far)

    logger.debug("Sample size: " + str(amount_sampled))
    return sample_indices, sample_values
