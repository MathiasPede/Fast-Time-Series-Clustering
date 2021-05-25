import numpy as np
import time
from math import log

import ftsc.sprl as sprl
from ftsc.solradm import solradm
from ftsc.aca import aca_symm
from ftsc.singular_values import calculate_best_relative_error_rank

from tests.plotting_utils import bar_plot, multiple_plot
from tests.tests_utils import create_cluster_problem, get_singular_values


def run_increasing_rank_test(data_name, func_name, max_rank=100, start_rank=10, step=5, test_size=5):
    # Fully sampled matrix to compare with
    cp = create_cluster_problem(data_name, func_name)
    sing_vals = get_singular_values(data_name, func_name)

    if max_rank > cp.cp_size():
        max_rank = cp.cp_size()

    error_size = 1 + int((max_rank-start_rank) / step)

    # Calculate the best possible errors based on the singular values
    best_errors = np.zeros(error_size)
    best_error_stds = np.zeros(error_size)

    for i in range(error_size):
        current_rank = start_rank + i * step

        for p in range(test_size):
            best_error = calculate_best_relative_error_rank(sing_vals, current_rank)
            best_errors[i] = best_error

    # Compute the approximations
    aca_averages, aca_stds, aca_sample_percentages = increasing_rank(aca_test, data_name, epsilon=0.001,
                                                                     max_rank=max_rank, start_rank=start_rank, step=step, test_size=test_size, copy_cp=cp, rank_factor=1)
    solrad1_averages, solrad1_stds, solrad1_sample_percentages = increasing_rank(solrad_test, data_name, epsilon=1.0,
                                                                                 max_rank=max_rank, start_rank=start_rank, step=step,
                                                                                 test_size=test_size, copy_cp=cp, rank_factor=1)
    solrad2_averages, solrad2_stds, solrad2_sample_percentages = increasing_rank(solrad_test, data_name, epsilon=2.0,
                                                                                 max_rank=max_rank, start_rank=start_rank, step=step,
                                                                                 test_size=test_size, copy_cp=cp, rank_factor=1)
    solrad4_averages, solrad4_stds, solrad4_sample_percentages = increasing_rank(solrad_test, data_name, epsilon=4.0,
                                                                                 max_rank=max_rank, start_rank=start_rank, step=step,
                                                                                 test_size=test_size, copy_cp=cp, rank_factor=1)

    # Plot the results
    data = np.array([aca_averages, solrad4_averages, solrad1_averages, best_errors])#, sprl_medians])
    xas = np.arange(start_rank, max_rank+step, step)
    yerr = np.array([aca_stds, solrad2_stds, solrad1_stds, best_error_stds])#, sprl_max_deviation])
    names = ("ACA", "SOLRADM $(\epsilon = 4.0)$", "SOLRADM $(\epsilon = 1.0)$", "SVD",  "SPRL")
    yname = "Relatieve fout (Frobenius norm)"
    title = "Relatieve fout van de benadering"

    bar_plot(data, xas, names, xname='Rank', yname=yname, title=title, yerr=yerr, yscale='linear')
    bar_plot((aca_sample_percentages, solrad4_sample_percentages, solrad1_sample_percentages), xas, ('ACA', "SOLRADM $(\epsilon = 4.0)$", "SOLRADM $(\epsilon = 1.0)$"), xname='Rang', yname='Percentage bemonsterd', title='Bemonsteringspercentage voor verschillende rangen', yerr=(None, None, None), yscale='linear')


def run_efficiency_test(data_name, func_name, max_rank=100, start_rank=10, step=5, test_size=5):
    # Fully sampled matrix to compare with
    cp = create_cluster_problem(data_name, func_name)

    if max_rank > cp.cp_size():
        max_rank = cp.cp_size()

    #error_size = 1 + int((max_rank-start_rank) / step)

    aca_averages, aca_stds, aca_sample_percentages = increasing_rank(aca_test, data_name, epsilon=0.001,
                                                                     max_rank=max_rank, start_rank=start_rank, step=step, test_size=test_size, copy_cp=cp, rank_factor=24)
    np.save("results/solrad1_averages_" + data_name, aca_averages)
    np.save("results/solrad1_percs_" + data_name, aca_sample_percentages)
    solrad1_averages, solrad1_stds, solrad1_sample_percentages = increasing_rank(solrad_test, data_name, epsilon=1.0,
                                                                                 max_rank=max_rank, start_rank=start_rank, step=step,
                                                                                 test_size=test_size, copy_cp=cp, rank_factor=1)
    np.save("results/aca_averages_" + data_name, solrad1_averages)
    np.save("results/aca_percs_" + data_name, solrad1_sample_percentages)
    solrad2_averages, solrad2_stds, solrad2_sample_percentages = increasing_rank(solrad_test, data_name, epsilon=2.0,
                                                                                 max_rank=max_rank, start_rank=start_rank, step=step,
                                                                                 test_size=test_size, copy_cp=cp, rank_factor=2)
    np.save("results/solrad2_averages_" + data_name, solrad2_averages)
    np.save("results/solrad2_percs_" + data_name, solrad2_sample_percentages)
    solrad4_averages, solrad4_stds, solrad4_sample_percentages = increasing_rank(solrad_test, data_name, epsilon=4.0,
                                                                                 max_rank=max_rank, start_rank=start_rank, step=step,
                                                                                 test_size=test_size, copy_cp=cp, rank_factor=4)
    np.save("results/solrad4_averages_" + data_name, solrad4_averages)
    np.save("results/solrad4_percs_" + data_name, solrad4_sample_percentages)

    multiple_plot((aca_averages, solrad1_averages, solrad2_averages, solrad4_averages),
                  (aca_sample_percentages, solrad1_sample_percentages, solrad2_sample_percentages, solrad4_sample_percentages),
                  ('ACA', "SOLRADM $(\epsilon = 1.0)$", "SOLRADM $(\epsilon = 2.0)$", "SOLRADM $(\epsilon = 4.0)$"),
                          xname='Bemonsteringspercentage', yname='Relatieve Fout', title='Efficientie van gebruik van monsters', yscale='log')


def show_results_efficiency_test(data_name):
    aca_averages=np.load("results/solrad1_averages_" + data_name + ".npy")
    aca_sample_percentages=np.load("results/solrad1_percs_" + data_name + ".npy")
    solrad1_averages=np.load("results/aca_averages_" + data_name + ".npy")
    solrad1_sample_percentages=np.load("results/aca_percs_" + data_name + ".npy")
    solrad2_averages=np.load("results/solrad2_averages_" + data_name + ".npy")
    solrad2_sample_percentages=np.load("results/solrad2_percs_" + data_name + ".npy")
    solrad4_averages=np.load("results/solrad4_averages_" + data_name + ".npy")
    solrad4_sample_percentages=np.load("results/solrad4_percs_" + data_name + ".npy")
    sprl_averages = np.load("results/sprl_averages_" + data_name + ".npy")
    sprl_sample_percentages = np.load("results/sprl_percs_" + data_name + ".npy")

    multiple_plot((aca_averages, solrad1_averages, solrad2_averages, solrad4_averages, sprl_averages),
                  (aca_sample_percentages, solrad1_sample_percentages, solrad2_sample_percentages,
                   solrad4_sample_percentages, sprl_sample_percentages),
                  ('ACA', "SOLRADM $(\epsilon = 1.0)$", "SOLRADM $(\epsilon = 2.0)$", "SOLRADM $(\epsilon = 4.0)$", "SPRL"),
                  xname='Bemonsteringspercentage', yname='Relatieve Fout', title='Efficientie van gebruik van monsters',
                  yscale='log')


def increasing_rank(test_func, data_name, epsilon=None, max_rank=100, start_rank=10, step=5, test_size=5, copy_cp=None, rank_factor=None):
    error_size = 1 + int((max_rank - start_rank) / step)
    errors = np.zeros((error_size, test_size))
    sample_perc = np.zeros((error_size, test_size))
    for i in range(error_size):
        current_rank = start_rank + i * step

        for p in range(test_size):
            print(test_func.__name__ + ": Computing for rank " + str(current_rank*rank_factor))
            start_aca = time.time()
            approx, cp = test_func(data_name, current_rank*rank_factor, epsilon=epsilon, copy_cp=copy_cp)
            sample_perc[i][p] = cp.percentage_sampled()
            end_aca = time.time()
            total_time = end_aca - start_aca
            print("Time for computing: " + str(total_time))
            print("Sample percentage: " + str(sample_perc[i][p]))

            error = cp.get_relative_error(approx)
            print("Rank " + str(current_rank*rank_factor) + ": ERROR: " + str(error))
            errors[i, p] = error
    averages = np.average(errors, axis=1)
    stds = np.std(errors, axis=1)
    sample_percentage_averages = np.average(sample_perc, axis=1)
    return averages, stds, sample_percentage_averages


def increasing_sample_percentage_sprl(data_name, func_name, start_perc=0.01, end_perc=0.22, step=0.01):
    # Fully sampled matrix to compare with
    cp = create_cluster_problem(data_name, func_name)

    n = cp.cp_size()
    sprl_sample_size = n * log(n)
    total_size = n * (n + 1) / 2
    sample_percentage = sprl_sample_size / total_size

    current_perc = start_perc
    sample_factor = start_perc / sample_percentage

    sample_percs = []
    errors = []

    while current_perc <= end_perc:
        print("SPRL : Computing for rank ")
        start_aca = time.time()
        approx, cp = sprl_test(data_name, 40, sample_factor=sample_factor, copy_cp=cp)
        sample_perc = cp.percentage_sampled()
        sample_percs.append(sample_perc)
        end_aca = time.time()
        total_time = end_aca - start_aca
        print("Time for computing: " + str(total_time))
        print("Sample percentage: " + str(sample_perc))

        error = cp.get_relative_error(approx)
        errors.append(error)
        print("Rank " + str(40) + ": ERROR: " + str(error))

        current_perc += step
        sample_factor = current_perc / sample_percentage

    errors_array = np.array(errors)
    sample_percs_array = np.array(sample_percs)
    np.save("results/sprl_averages_"+str(data_name), errors_array)
    np.save("results/sprl_perc_results_"+str(data_name), sample_percs_array)
    return errors_array, None, sample_percs_array


def get_number_of_tests(start_rank, max_rank, step):
    i = 0
    current = start_rank
    while current < max_rank:
        i += 1
        current += step
    return i


def aca_test(data_name, max_rank, epsilon=None, copy_cp=None):
    if copy_cp is None:
        cp = create_cluster_problem(data_name, "msm")
    else:
        cp = copy_cp.make_non_sampled_copy()
    approx = aca_symm(cp, tolerance=epsilon, max_rank=max_rank)
    return approx, cp


def solrad_test(data_name, rank, epsilon=None, copy_cp=None):
    if copy_cp is None:
        cp = create_cluster_problem(data_name, "msm")
    else:
        cp = copy_cp.make_non_sampled_copy()
    approx = solradm(cp, rank=rank, epsilon=epsilon)
    return approx, cp


def sprl_test(data_name, rank, sample_factor=None, copy_cp=None):
    if copy_cp is None:
        cp = create_cluster_problem(data_name, "msm")
    else:
        cp = copy_cp.make_non_sampled_copy()
    distances_with_zero = sprl.compute_distances_with_zero(cp.series, cp.compare)
    temp = cp.solved_matrix
    print("Converting to similarity")
    cp.solved_matrix = sprl.convert_distance_matrix_to_similarity_matrix(cp.solved_matrix, distances_with_zero)
    approx = sprl.convert_similarity_matrix_to_distance_matrix(sprl.spiral(cp, rank, sample_factor=sample_factor), distances_with_zero)
    cp.solved_matrix = temp
    return approx, cp


if __name__ == "__main__":
    name = "ECG5000"
    func_name = "dtw"
    max_rank = 10

    #run_increasing_rank_test(name, func_name, max_rank=50, start_rank=10, step=20, test_size=1)
    #run_efficiency_test(name, func_name, max_rank=9, start_rank=5, step=2, test_size=1)
    #run_all_datasets_better_than_accuracy_test(func_name, [0.1, 0.05, 0.02, 0.01], start_index=0)
    #increasing_sample_percentage_sprl(name, func_name)
    show_results_efficiency_test(name)
