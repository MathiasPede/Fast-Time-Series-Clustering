from solrad import compute_probabilities
import numpy as np
import cluster_problem as cp
import scipy.stats
import random as rnd



def get_c_number(real_probs, probabilities):
    compared = np.divide(probabilities, real_probs)
    c = np.min(compared)
    return c


def get_real_probabilities(matrix):
    size = len(matrix)
    probabilities = np.zeros(size)

    for i in range(size):
        norm = np.linalg.norm(matrix[:,i])
        probabilities[i] = norm * norm

    probabilities = probabilities / np.sum(probabilities)
    return probabilities


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """
    # calculate m
    m = (p + q) / 2
    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)
    return distance


def compute_probabilities(matrix):
    """
    Calculate the probabilities for the sampling algorithm by approximating the norms of the
    rows using a reference element of the matrix.
    :param cp: ClusterProblem (contains the matrix)
    :return: numepy array with probabilities (normalized already)
    """
    # Initialize list of probabilities
    size = len(matrix)
    arr = np.zeros(size)
    ref = rnd.randint(0, size - 1)

    # Choose a random time series as reference
    ref_row = matrix[ref]
    squared = np.square(ref_row)
    average_distance_to_ref = np.average(squared)
    arr = squared + average_distance_to_ref
    result = arr / np.sum(arr)

    return result


if __name__ == '__main__':
    from data_loader import get_all_dataset_names, compute_distance_matrix_msm, read_train_and_test_data, compute_distance_matrix_dtw, compute_distance_matrix_ed, take_submatrix_matrix
    from msm import msm_fast
    from plotter import histogram_plot, scatter_plot, multiple_scatter_plot

    # name = "ElectricDevices"
    # test_size = 5
    #
    # data = read_train_and_test_data(name)
    # size = len(data)
    # matrix = compute_distance_matrix_msm(data, name)
    # cluster_problem = cp.ClusterProblem(data, msm_fast, solved_matrix=matrix)
    #
    # real_probs = get_real_probabilities(matrix)
    # uniform = np.ones(size, dtype=float) / size
    # c_numbers_uniform = np.divide(uniform, real_probs)
    #
    # test_cs = np.zeros(test_size, dtype=float)
    # ref_norms = np.zeros(test_size, dtype=float)
    #
    # for i in range(test_size):
    #     ref_index = rnd.randint(0, size - 1)
    #     #norms_estimated = compute_norm_estimations(cluster_problem, ref_index)
    #     #probs_estimated = norms_estimated / np.sum(norms_estimated)
    #     probs_estimated = compute_probabilities(cluster_problem, rows_amount=10)
    #     c_numbers = np.divide(probs_estimated, real_probs)
    #     c = np.min(c_numbers)
    #     print(c)
    #     #test_cs[i] = c
    #     #ref_norms[i] = np.linalg.norm(cluster_problem.sample_row(ref_index))
    #     multiple_scatter_plot([c_numbers, c_numbers_uniform], [real_probs, real_probs], ['Schatting', 'Uniform'], yscale='linear', title="Correlatie kans schatting SOLRADM vs uniform", yname='Geschatte kans / Correcte kans', xname='Correcte kans')

    #scatter_plot(test_cs, ref_norms, title="Invloed norm referentie rij op inschatting kansverdeling",
    #             yname="c factor", xname='Norm referentie', yscale='log')
    #average = all_probs / test_size
    #compared = np.divide(average, real_probs)
    #histogram_plot(real_probs)
    #histogram_plot(compared)


    # all_datasets = get_all_dataset_names()
    # size = len(all_datasets)
    # test_size = 1000
    #
    # results = np.zeros((size,test_size))
    # probability_distance = np.zeros((size, test_size))
    #
    # for i in range(size):
    #     name = all_datasets[i]
    #     print("Testing dataset: " + name)
    #
    #     data = read_train_and_test_data(name)
    #     matrix = compute_distance_matrix_dtw(data, name)
    #     cluster_problem = cp.ClusterProblem(data, msm_fast, solved_matrix=matrix)
    #
    #     real_probs = get_real_probabilities(matrix)
    #
    #     for j in range(test_size):
    #         probs_estimated = compute_probabilities(cluster_problem, rows_amount=1)
    #         results[i,j] = get_c_number(real_probs, probs_estimated)
    #         #probability_distance[i,j] = jensen_shannon_distance(real_probs, probs_estimated)
    #
    #     average = np.sum(results[i,:]) / test_size
    #     std = np.std(results[i,:])
    #
    #     #average_distance = np.sum(probability_distance[i,:]) / test_size
    #
    #     print("AVERAGE: " + str(average))
    #     print("STD: " + str(std))
    #     #print('JS_Distance: ' + str(average_distance))
    #
    # np.save("results/norm_estimation_dtw_c_test_1_row_" + str(test_size), results)

    # #np.save("results/norm_estimation_jcdistance_test_" + str(test_size), probability_distance)

    # all_datasets = get_all_dataset_names()
    # size = len(all_datasets)
    #
    # set_sizes = np.zeros(size)
    #
    # for i in range(size):
    #     name = all_datasets[i]
    #     data = read_train_and_test_data(name)
    #     print(name)
    #     set_sizes[i] = len(data)
    #
    #results = np.load("results/norm_estimation_dtw_c_test_10_rows_1000.npy")
    # relative_errors = np.load("results/triangle_relative_errors.npy")
    # indices = [9,16,17,19,21,26,27,28,29,37,42,45,46,48,49,52,53,65,74,76,81]
    # errors_filtered = np.delete(relative_errors, indices)
    #
    #average = np.average(results, axis=1)
    # average_filtered = np.delete(average, indices, axis=0)
    #
    # scatter_plot(average_filtered, errors_filtered, yscale='linear', marker="o", yname='c factor', xname='Afstand tot metrische afstandsmatrix', title='Impact driehoeksongelijkheid op schatting normen')
    # scatter_plot(average, set_sizes, yscale='linear', marker="o", yname='c factor', xname='Grootte matrix', title='Impact grootte matrix op schatting normen')
    #total_average = np.average(average)
    #std = np.std(results, axis=0)
    #total_std = np.average(std)

    name = "Crop"
    test_size = 250
    data = read_train_and_test_data(name)
    full_size = len(data)

    full_matrix = compute_distance_matrix_msm(data, name)
    sizes = np.zeros(test_size, dtype=int)
    results = np.zeros(test_size, dtype=float)

    for i in range(test_size):
        print("Test " + str(i))
        sizes[i] = rnd.randint(0, full_size - 1)
        indices = rnd.sample(range(0, full_size), sizes[i])
        submatrix = take_submatrix_matrix(full_matrix, indices)
        real_probs = get_real_probabilities(submatrix)
        probs_estimated = compute_probabilities(submatrix)
        c_numbers = np.divide(probs_estimated, real_probs)
        c = np.min(c_numbers)
        results[i] = c

    scatter_plot(results, sizes, yscale='linear', marker="o", yname='c factor', xname='Grootte matrix', title='Impact matrixgrootte op schatting normen')




