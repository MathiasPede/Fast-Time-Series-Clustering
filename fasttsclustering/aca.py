import numpy as np
import data_loader as dl
from math import sqrt
from math import floor
import random as rnd
import cluster_problem as cp


def aca_body(cp, max_rank=None, sample_size=None, tolerance=0.05, debug=False, seed=None, start_index=None):
    if not max_rank:
        max_rank = cp.cp_size()
    if seed:
        rnd.seed(seed)

    rows = np.zeros(shape=(max_rank, cp.cp_size()))
    cols = np.zeros(shape=(cp.cp_size(), max_rank))

    samples = generate_samples_student_distribution(cp)
    sample_size = len(samples)
    restartable_samples = samples

    # Threshold value is tolerance times the smallest sampled value
    initial_average = np.average(np.square(samples[:, 2]))
    best_remaining_average = initial_average

    # Start random row
    if start_index and 0 <= start_index <= cp.cp_size():
        print("Given start index " + str(start_index))
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

    while m < max_rank - 1 and not stopcrit:
        if debug:
            print("Iteration " + str(m) + " New pivot: " + str(pivot_index))
        if debug:
            print("The indices so far: " + str(indices))

        # Calculate the current approximation for row of pivot
        row_approx = np.zeros(cp.cp_size(), dtype=float)
        for i in range(m):
            row_approx = np.add(row_approx, rows[i, :] * cols[pivot_index, i])

        # Find new w vector
        new_row = np.subtract(cp.sample_row(pivot_index), row_approx)
        # Find delta
        new_delta_index = np.argmax(np.abs(new_row))
        new_delta = new_row[new_delta_index]

        # Check whether the delta is within the tolerance
        if abs(new_delta) < max_residu - 0.001:
            # Switch to the pivot to the sample with the largest remaining value
            index_sample_max = np.where(np.abs(restartable_samples[:, 2]) == max_residu)[0][0]
            pivot_index = int(restartable_samples[index_sample_max][0] + 0.00001)
        else:
            indices.append(pivot_index)
            rows[m, :] = new_row

            # Calculate the column w vector
            col_approx = np.zeros(cp.cp_size(), dtype=float)
            for i in range(m):
                col_approx = np.add(col_approx, rows[i, :] * cols[pivot_index, i])

            new_col = (1.0 / new_delta) * np.subtract(cp.sample_col(new_delta_index), col_approx)
            cols[:, m] = new_col

            for j in range(sample_size):
                x = int(samples[j][0] + 0.00001)
                y = int(samples[j][1] + 0.00001)
                samples[j][2] = samples[j][2] - cols[x, m] * rows[m, y]

            pivot_indices_in_samples = np.where(samples[:, 0] == pivot_index)[0]
            if deleted_indices.size == 0:
                deleted_indices = pivot_indices_in_samples
            else:
                deleted_indices = np.concatenate((deleted_indices, pivot_indices_in_samples), axis=0)

            restartable_samples = np.delete(samples, deleted_indices, axis=0)
            max_residu = np.max(np.abs(restartable_samples[:, 2]))

            remaining_average = np.average(np.square(samples[:, 2]))
            stopcrit = (sqrt(remaining_average) < sqrt(initial_average) * tolerance)

            if remaining_average < best_remaining_average:
                best_remaining_average = remaining_average
                best_m = m

            if m > 5 and remaining_average > initial_average:
                estimated_error = sqrt(best_remaining_average) / sqrt(initial_average)
                print("Diverged, best approximation had error: " + str(estimated_error))
                return rows, cols, best_m, estimated_error

            if debug:
                if stopcrit:
                    print("The stopcriterion was achieved")

            row_without_already_sampled_indices = np.delete(new_row, indices, axis=0)
            new_max = np.max(np.abs(row_without_already_sampled_indices))
            pivot_index = np.where(np.abs(new_row) == new_max)[0][0]

            # loop iterator
            m += 1
            full_approx = calc_matrix_approx(cp, rows, cols, m)

    # result is the sum of rank 1 matrices (u * v)
    if debug:
        print("The indices taken for this approximation are: " + str(indices))
        print("ACA resulting approximation with rank " + str(m - 1))
        print("Best rank was: " + str(best_m))

    estimated_error = sqrt(best_remaining_average) / sqrt(initial_average)
    return rows, cols, best_m, estimated_error


def aca_relaxed_restart(cp, tolerance=0.05, max_restarts=10, max_rank=None, start_index=None, seed=None, relax_factor=1.1, debug=False):
    min_error = 1.0
    best_rows = None
    best_cols = None
    best_m = None
    for i in range(max_restarts):
        rows, cols, m, error = aca_body(cp, max_rank=max_rank, tolerance=tolerance, seed=seed, start_index=start_index, debug=debug)
        if error < min_error:
            best_rows, best_cols, best_m, min_error = rows, cols, m, error
            if min_error < tolerance:
                break
        tolerance *= relax_factor
        print("Relaxing tolerance to: " + str(tolerance))
        if min_error < tolerance:
            print("This tolerance was achieved earlier")
            break
    print("ACA resulting approximation with rank " + str(best_m))
    print("Estimated relative error: " + str(min_error))
    return calc_matrix_approx(cp, best_rows, best_cols, best_m)


def aca_symmetric_body(cp, max_rank=None, tolerance=0.05, debug=False, seed=None,
                                      start_index=None):
    if not max_rank or max_rank > cp.cp_size():
        max_rank = cp.cp_size()
    if seed:
        rnd.seed(seed)

    rows = np.zeros(shape=(max_rank, cp.cp_size()))
    deltas = np.zeros(max_rank)

    samples = generate_samples_student_distribution(cp)
    sample_size = len(samples)

    # Threshold value is tolerance times the smallest sampled value
    initial_average = np.average(np.square(samples[:, 2]))
    best_remaining_average = initial_average

    # Start random row
    if start_index and 0 <= start_index <= cp.cp_size():
        print("Given start index " + str(start_index))
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
    restartable_samples = samples

    while m < max_rank - 1 and not stopcrit:
        if debug:
            print("Iteration " + str(m) + " New pivot: " + str(pivot_index))
        if debug:
            print("The indices so far: " + str(indices))

        # Calculate the current approximation for row of pivot
        approx = np.zeros(cp.cp_size())
        for i in range(m):
            approx = np.add(approx, rows[i, :] * rows[i, pivot_index] * (1.0 / deltas[i]))

        # Find new w vector
        new_row = np.subtract(cp.sample_row(pivot_index), approx)
        # Save delta

        new_max_index = np.argmax(np.abs(new_row))
        new_max = new_row[new_max_index]
        new_delta = new_row[pivot_index]
        if new_delta == 0:
            new_delta = new_max
            if new_delta == 0.0:
                index_sample_max = np.where(np.abs(restartable_samples[:, 2]) == max_residu)[0][0]
                pivot_index = int(restartable_samples[index_sample_max][0] + 0.00001)
                continue

        indices.append(pivot_index)
        rows[m, :] = new_row
        deltas[m] = new_delta

        for j in range(sample_size):
            x = int(samples[j][0] + 0.00001)
            y = int(samples[j][1] + 0.00001)
            samples[j][2] = samples[j][2] - (1.0 / deltas[m]) * rows[m, y] * rows[m, x]

        pivot_indices_in_row_of_samples = np.where(samples[:, 0] == pivot_index)[0]
        pivot_indices_in_col_of_samples = np.where(samples[:, 1] == pivot_index)[0]
        pivot_indices_in_samples = np.concatenate((pivot_indices_in_row_of_samples, pivot_indices_in_col_of_samples))
        if deleted_indices.size == 0:
            deleted_indices = pivot_indices_in_samples
        else:
            deleted_indices = np.concatenate((deleted_indices, pivot_indices_in_samples), axis=0)

        restartable_samples = np.delete(samples, deleted_indices, axis=0)
        if restartable_samples.size == 0:
            max_residu = 0
        else:
            max_residu = np.max(np.abs(restartable_samples[:, 2]))

        remaining_average = np.average(np.square(samples[:, 2]))
        stopcrit = (sqrt(remaining_average) < sqrt(initial_average) * tolerance)

        if remaining_average < best_remaining_average:
            best_remaining_average = remaining_average
            best_m = m

        if m > 5 and m > best_m + max(floor(0.02 * cp.cp_size()), 100):
            estimated_error = sqrt(best_remaining_average) / sqrt(initial_average)
            print("No improvement for 100 ranks, current rank: " + str(m))
            print("Best approximation had error: " + str(estimated_error))
            return rows, deltas, best_m, estimated_error

        new_row_abs = np.abs(new_row)
        row_without_already_sampled_indices = np.delete(new_row_abs, indices, axis=0)
        new_max = np.max(row_without_already_sampled_indices)
        pivot_index = np.where(new_row_abs == new_max)[0][0]

        # Check whether the max of the row is smaller than the max residu from the samples
        if abs(new_max) < max_residu - 0.001:
            # Switch to the pivot to the sample with the largest remaining value
            index_sample_max = np.where(np.abs(restartable_samples[:, 2]) == max_residu)[0][0]
            pivot_index = int(restartable_samples[index_sample_max][0] + 0.00001)

        m += 1

    if stopcrit:
        print("stopcrit: Approximated error: " + str(sqrt(best_remaining_average) / sqrt(initial_average)))

    estimated_error = sqrt(best_remaining_average) / sqrt(initial_average)
    return rows, deltas, best_m, estimated_error


def aca_symm(cp, tolerance=0.05, max_rank=None, start_index=None, seed=None, debug=False):
    rows, deltas, m, error = aca_symmetric_body(cp, max_rank=max_rank, tolerance=tolerance, seed=seed, start_index=start_index, debug=debug)
    if True:
        print("ACA resulting approximation with rank " + str(m))
    matrix = calc_symmetric_matrix_approx(rows, deltas, cp.cp_size(), m)
    np.fill_diagonal(matrix, 0)
    return matrix


def calc_symmetric_matrix_approx(rows, deltas, size, iters):
    cols = np.transpose(rows[0:iters]).copy()
    for i in range(iters):
        cols[:,i] *= 1.0 / deltas[i]
    result = np.matmul(cols, rows[0:iters])
    return result


def generate_samples(amount, cp):
    samples = np.zeros(shape=(amount, 3), dtype=float)
    size = cp.cp_size()
    for i in range(amount):
        x, y = 0, 0
        while x == y:
            x = rnd.randint(0, size - 1)
            y = rnd.randint(0, size - 1)
        samples[i][0] = x
        samples[i][1] = y
        samples[i][2] = cp.sample(x, y)
    return samples


def generate_samples_student_distribution(cp):
    amount_sampled = 0
    average_so_far = 0
    std_so_far = 0
    t = 3.39
    tolerance = 1000000000000000
    indices = []
    samples = []
    size = cp.cp_size()

    samples = np.array([], dtype=float)
    while tolerance > 0.01 or amount_sampled < 2 * cp.cp_size():
        iteration_samples = np.zeros(shape=(size, 3), dtype=float)
        for i in range(size):
            x = i
            y = i
            while x == y:
                y = rnd.randint(0, size - 1)
            iteration_samples[i][0] = x
            iteration_samples[i][1] = y
            iteration_samples[i][2] = cp.sample(x, y)

        if samples.size == 0:
            samples = iteration_samples
        else:
            samples = np.concatenate((samples, iteration_samples))

        squared = np.square(samples[:, 2])
        average_so_far = np.mean(squared)
        std_so_far = np.std(squared)
        amount_sampled += size
        if amount_sampled > cp.cp_size()*cp.cp_size():
            break

        tolerance = (t * std_so_far) / (sqrt(amount_sampled) * average_so_far)

    print("Sample size: " + str(amount_sampled))

    return samples


def calc_matrix_approx(cp, rows, cols, iters):
    result = np.zeros(shape=(cp.cp_size(), cp.cp_size()))
    for i in range(iters):
        result = np.add(result, np.outer(cols[:, i], rows[i, :]))

    return result


def make_symmetrical(matrix):
    """
    Converts an approximation generated by ACA (which isn't symmetrical) to a symmetrical matrix by
    taking the average between the entries at i,j and j,i
    """
    size = len(matrix)
    result = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            average = (matrix[i, j] + matrix[j, i]) / 2.0
            result[i, j] = average
            result[j, i] = average
    return result


if __name__ == "__main__":
    from dtaidistance import dtw
    from singular_values import calculate_best_relative_error_rank
    """
    Start testing
    """
    data_name = "Crop"
    rank = None
    debug = False
    start_index = None
    func_name = "msm"
    tolerance = 0.02
    index = 0

    names = dl.get_all_dataset_names()
    size = len(names)
    try:
        errors = np.load("results/all_sets_aca_" + str(tolerance) + "_errors.npy")
        percs = np.load("results/all_sets_aca_" + str(tolerance) + "_percs.npy")
    except:
        errors = np.zeros(size)
        percs = np.zeros(size)

    sample_factors = np.zeros(size)
    for i in range(index, size):
         data_name = names[i]
    #     print("TESTING DATASET: " + data_name)
    #     # Read data from csv
         my_data = dl.read_train_and_test_data(data_name, debug=debug)
    #     print("Size: " + str(len(my_data)))
    #
         solved_matrix = dl.load_array(data_name, func_name)
    #
    #     # Create the problem class with internally the samplable matrix
         problem = cp.ClusterProblem(my_data, dtw.distance_fast, solved_matrix=solved_matrix)
    #
    #     approx = aca_symm(problem, tolerance=tolerance, debug=debug, start_index=start_index)
    #
    #     percs[i] = problem.percentage_sampled()
    #     print("Percentage sampled = " + str(percs[i]))
    #
         sample_factors[i] = percs[i] * problem.get_max_sample_amount() / problem.cp_size()
    #     print("Sample factor = " + str(sample_factors[i]))
    #
    #     errors[i] = problem.get_relative_error(approx)
    #     print("Error = " + str(errors[i]))
    #
    #     print(" & " + str(round(errors[i], 4)) + " & " + str(round(100*percs[i],2)) + " & " + str(round(sample_factors[i], 1)))
    #
    #     np.save("results/all_sets_aca_" + str(tolerance) + "_percs", percs)
    #     np.save("results/all_sets_aca_" + str(tolerance) + "_errors", errors)

    average = np.average(errors)
    std = np.std(errors)

    average_f = np.average(sample_factors)
    std_f = np.std(sample_factors)