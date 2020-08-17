import numpy as np
import random as rnd

from ftsc.solradm import compute_probabilities
from ftsc.triangle_fixing import distance_to_metric_space

from tests.tests_utils import create_cluster_problem, take_submatrix_matrix
from tests.plotting_utils import multiple_scatter_plot, scatter_plot, text_scatter_plot
from tests.tests_utils import get_all_test_dataset_names


# Utils for tests
def get_c_number(real_probs, probabilities):
    compared = np.divide(probabilities, real_probs)
    c = np.min(compared)
    return c


def get_real_probabilities(matrix):
    size = len(matrix)
    probabilities = np.zeros(size)

    for i in range(size):
        norm = np.linalg.norm(matrix[:, i])
        probabilities[i] = norm * norm

    probabilities = probabilities / np.sum(probabilities)
    return probabilities


def compute_probabilities_with_index(matrix, index):
    ref_row = matrix[index, :]
    squared = np.square(ref_row)
    average_distance_to_ref = np.average(squared)
    arr = squared + average_distance_to_ref
    result = arr / np.sum(arr)
    return result


# TESTS

def estimate_probabilities_test(name, func_name, test_size, row_amount=1):
    cp = create_cluster_problem(name, func_name)
    matrix = cp.sample_full_matrix()
    size = cp.cp_size()

    real_probs = get_real_probabilities(matrix)
    uniform = np.ones(size, dtype=float) / size
    c_numbers_uniform = np.divide(uniform, real_probs)

    for i in range(test_size):
        probs_estimated = compute_probabilities(cp, rows_amount=row_amount)
        c_numbers = np.divide(probs_estimated, real_probs)
        multiple_scatter_plot([c_numbers, c_numbers_uniform], [real_probs, real_probs], ['Schatting', 'Uniform'],
                              yscale='linear', title="Correlatie kans schatting SOLRADM vs uniform",
                              yname='Geschatte kans / Correcte kans', xname='Correcte kans')


def norm_reference_test(name, func_name, test_size):
    cp = create_cluster_problem(name, func_name)
    matrix = cp.sample_full_matrix()
    size = cp.cp_size()
    real_probs = get_real_probabilities(matrix)

    test_cs = np.zeros(test_size, dtype=float)
    ref_norms = np.zeros(test_size, dtype=float)

    for i in range(test_size):
        ref_index = rnd.randint(0, size)
        probs_estimated = compute_probabilities_with_index(matrix, ref_index)
        c_numbers = np.divide(probs_estimated, real_probs)
        c = np.min(c_numbers)
        test_cs[i] = c
        ref_norms[i] = np.linalg.norm(cp.sample_row(ref_index))

    scatter_plot(test_cs, ref_norms, title="Invloed norm referentie rij op inschatting kansverdeling",
                 yname="c factor", xname='Norm referentie', yscale='log')


def average_and_std_all_datasets_test(func_name):
    all_datasets = get_all_test_dataset_names()
    size = len(all_datasets)
    test_size = 1000

    all_averages = np.zeros(size)
    all_stds = np.zeros(size)

    for i in range(size):
        name = all_datasets[i]
        print("Testing dataset: " + name)

        cp = create_cluster_problem(name, func_name)
        matrix = cp.sample_full_matrix()

        real_probs = get_real_probabilities(matrix)

        results = np.zeros(test_size)
        for j in range(test_size):
            probs_estimated = compute_probabilities(cp, rows_amount=1)
            results[j] = get_c_number(real_probs, probs_estimated)

        all_averages[i] = np.average(results)
        all_stds[i] = np.std(results)

        print("AVERAGE: " + str(all_averages[i]))
        print("STD: " + str(all_stds[i]))

    total_average = np.average(all_averages)
    total_std = np.average(all_stds)

    print("TOTAL AVERAGE: " + str(total_average))
    print("TOTAL STD: " + str(total_std))

    return all_averages, all_stds


def compare_average_c_factor_with_distance_to_metric_space_test(averages, max_size=1500):
    all_datasets = get_all_test_dataset_names()
    size = len(all_datasets)

    errors = []
    names = []
    average_cs = []

    for i in range(size):
        name = all_datasets[i]
        cp = create_cluster_problem(name, "dtw")
        if cp.cp_size() < max_size:
            matrix = cp.sample_full_matrix()
            errors.append(distance_to_metric_space(matrix))
            names.append(name)
            average_cs.append(averages[i])

    text_scatter_plot(average_cs, errors, yscale='linear', marker="o", yname='c factor',
                 xname='Afstand tot metrische afstandsmatrix', title='Impact driehoeksongelijkheid op schatting normen',
                 names=names, regression=True)


def submatrix_test(name, func_name):
    test_size = 250
    cp = create_cluster_problem(name, func_name)
    full_matrix = cp.sample_full_matrix()
    full_size = cp.cp_size()
    sizes = np.zeros(test_size, dtype=int)
    results = np.zeros(test_size, dtype=float)
    for i in range(test_size):
        sizes[i] = rnd.randint(0, full_size - 1)
        indices = rnd.sample(range(0, full_size), sizes[i])
        submatrix = take_submatrix_matrix(full_matrix, indices)
        real_probs = get_real_probabilities(submatrix)
        probs_estimated = compute_probabilities(submatrix)
        c_numbers = np.divide(probs_estimated, real_probs)
        c = np.min(c_numbers)
        results[i] = c
    scatter_plot(results, sizes, yscale='linear', marker="o", yname='c factor', xname='Grootte matrix', title='Impact matrixgrootte op schatting normen')


if __name__ == '__main__':
    name = "ElectricDevices"
    func_name = "msm"

    print("TEST 1: Probabilities basis test")
    estimate_probabilities_test(name, func_name, 5)

    print("TEST 2: c vs norm reference test")
    norm_reference_test(name, func_name, 2000)

    print("TEST 3: Average over all datasets test")
    average_and_std_all_datasets_test(func_name)

    print("TEST 4: c vs distance to metric space test")
    compare_average_c_factor_with_distance_to_metric_space_test()

    print("TEST 5: Size impact test")
    submatrix_test(name, func_name)