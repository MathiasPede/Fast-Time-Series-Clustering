import data_loader as dl
import aca
import solrad
import sprl
import cluster_problem
import distance_functions as df
import numpy as np
import best_low_rank_approximation as blra
import time
from dtaidistance.dtw import distance_fast as dtw_fast
from msm import msm_fast
from ed import ed_fast
from plotter import bar_plot
from plotter import scatter_plot
from plotter import multiple_scatter_plot, multiple_plot
from singular_values import calculate_best_relative_error_rank, get_first_rank_relative_error_below
from math import floor, log


def run_increasing_rank_test(data_name, func_name, max_rank=100, start_rank=10, step=5, test_size=5, debug=False):
    # Fully sampled matrix to compare with
    cp = make_cluster_problem(data_name, func_name)

    if max_rank > cp.cp_size():
        max_rank = cp.cp_size()

    error_size = 1 + int((max_rank-start_rank) / step)

    best_errors = np.zeros(error_size)
    best_error_stds = np.zeros(error_size)

    for i in range(error_size):
        current_rank = start_rank + i * step

        for p in range(test_size):
            best_error = calculate_best_relative_error_rank(data_name, func_name, current_rank)
            best_errors[i] = best_error


    aca_averages, aca_stds, aca_sample_percentages = increasing_rank(test_aca, data_name, epsilon=0.001,
                    max_rank=max_rank, start_rank=start_rank, step=step, test_size=test_size, copy_cp=cp, rank_factor=1)
    solrad1_averages, solrad1_stds, solrad1_sample_percentages = increasing_rank(test_solrad, data_name, epsilon=1.0,
                                                                     max_rank=max_rank, start_rank=start_rank, step=step,
                                                                     test_size=test_size, copy_cp=cp, rank_factor=1)
    solrad2_averages, solrad2_stds, solrad2_sample_percentages = increasing_rank(test_solrad, data_name, epsilon=2.0,
                                                                     max_rank=max_rank, start_rank=start_rank, step=step,
                                                                     test_size=test_size, copy_cp=cp, rank_factor=1)
    solrad4_averages, solrad4_stds, solrad4_sample_percentages = increasing_rank(test_solrad, data_name, epsilon=4.0,
                                                                     max_rank=max_rank, start_rank=start_rank, step=step,
                                                                     test_size=test_size, copy_cp=cp, rank_factor=1)

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
    cp = make_cluster_problem(data_name, func_name)

    if max_rank > cp.cp_size():
        max_rank = cp.cp_size()

    error_size = 1 + int((max_rank-start_rank) / step)

    best_errors = np.zeros(error_size)

    for i in range(error_size):
        current_rank = start_rank + i * step

        best_error = calculate_best_relative_error_rank(data_name, func_name, current_rank)
        best_errors[i] = best_error

    aca_averages, aca_stds, aca_sample_percentages = increasing_rank(test_aca, data_name, epsilon=0.001,
                    max_rank=max_rank, start_rank=start_rank, step=step, test_size=test_size, copy_cp=cp, rank_factor=24)
    np.save("results/solrad1_averages_" + data_name, aca_averages)
    np.save("results/solrad1_percs_" + data_name, aca_sample_percentages)
    solrad1_averages, solrad1_stds, solrad1_sample_percentages = increasing_rank(test_solrad, data_name, epsilon=1.0,
                                                                     max_rank=max_rank, start_rank=start_rank, step=step,
                                                                     test_size=test_size, copy_cp=cp, rank_factor=1)
    np.save("results/aca_averages_" + data_name, solrad1_averages)
    np.save("results/aca_percs_" + data_name, solrad1_sample_percentages)
    solrad2_averages, solrad2_stds, solrad2_sample_percentages = increasing_rank(test_solrad, data_name, epsilon=2.0,
                                                                     max_rank=max_rank, start_rank=start_rank, step=step,
                                                                     test_size=test_size, copy_cp=cp, rank_factor=2)
    np.save("results/solrad2_averages_" + data_name, solrad2_averages)
    np.save("results/solrad2_percs_" + data_name, solrad2_sample_percentages)
    solrad4_averages, solrad4_stds, solrad4_sample_percentages = increasing_rank(test_solrad, data_name, epsilon=4.0,
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


def run_better_than_accuracies_test(data_name, func_name, test_func, accuracies, epsilon=0.001, rank_factor=1.0, copy_cp=None):
    max_rank = floor(copy_cp.cp_size() * 0.05 * rank_factor)
    size_accuracies = len(accuracies)

    ranks = np.zeros(size_accuracies)
    sample_percentages = np.zeros(size_accuracies)

    current_rank = 2

    print("Testing: " + test_func.__name__)
    for i in range(len(accuracies)):
        current_rank = max(current_rank, get_first_rank_relative_error_below(data_name, func_name, accuracies[i]))
        if current_rank > max_rank:
            print("Accuracy " + str(accuracies[i]) + "can only be achieved with rank " + str(current_rank))
            break
        while True:
            print("Testing rank " + str(current_rank))
            approx, func_cp = test_func(data_name, current_rank, epsilon=epsilon, copy_cp=copy_cp)
            error = copy_cp.get_relative_error(approx)
            perc = func_cp.percentage_sampled()
            if error < accuracies[i]:
                ranks[i] = current_rank
                sample_percentages[i] = perc
                print("ERROR: " + str(error) + " Rank: " + str(current_rank) + " PERC: " + str(round(sample_percentages[i], 3)))
                current_rank = floor(current_rank * 1.2) + 1
                break
            current_rank = floor(current_rank*1.2) + 1
            if perc > 0.5:
                print("Last sample percentage: " + str(round(perc, 3)))
                print("Max rank achieved: " + str(current_rank))
                print("Last error: " + str(error))
                break
        if perc > 0.5:
            break

    return ranks, sample_percentages


def run_all_datasets_better_than_accuracy_test(func_name, accuracies, start_index=0):
    all_datasets = dl.get_all_dataset_names()
    size = len(all_datasets)
    acc_size = len(accuracies)

    try:
        aca_rank_results = np.load("results/aca_rank_results_" + str(func_name)+".npy")
        solrad1_rank_results = np.load("results/solrad1_rank_results_" + str(func_name)+".npy")
        solrad2_rank_results = np.load("results/solrad2_rank_results_" + str(func_name)+".npy")

        aca_perc_results = np.load("results/aca_perc_results_" + str(func_name)+".npy")
        solrad1_perc_results = np.load("results/solrad1_perc_results_" + str(func_name)+".npy")
        solrad2_perc_results = np.load("results/solrad2_perc_results_" + str(func_name)+".npy")
        print("Loaded")
    except Exception:
        aca_rank_results = np.zeros((size, acc_size))
        solrad1_rank_results = np.zeros((size, acc_size))
        solrad2_rank_results = np.zeros((size, acc_size))

        aca_perc_results = np.zeros((size, acc_size))
        solrad1_perc_results = np.zeros((size, acc_size))
        solrad2_perc_results = np.zeros((size, acc_size))

    for i in range(start_index, size):
        name = all_datasets[i]
        # Fully sampled matrix to compare with
        cp = make_cluster_problem(name, func_name)
        if cp.cp_size() < 2000:
            print("Size too small: " + name)
            continue
        print("Testing index " + str(i) + " (= " + name + ")")

        aca_rank_results[i], aca_perc_results[i] = run_better_than_accuracies_test(name, func_name, test_aca, accuracies,
                                                                     epsilon=0.0001, rank_factor=12, copy_cp=cp)
        solrad1_rank_results[i], solrad1_perc_results[i] = run_better_than_accuracies_test(name, func_name, test_solrad,
                                                                             accuracies, epsilon=2.0,
                                                                             rank_factor=2, copy_cp=cp)
        solrad2_rank_results[i], solrad1_perc_results[i] = run_better_than_accuracies_test(name, func_name, test_solrad,
                                                                             accuracies, epsilon=4.0,
                                                                             rank_factor=4, copy_cp=cp)

        np.save("results/aca_rank_results_" + str(func_name), aca_rank_results)
        np.save("results/solrad1_rank_results_" + str(func_name), solrad1_rank_results)
        np.save("results/solrad2_rank_results_" + str(func_name), solrad2_rank_results)

        np.save("results/aca_perc_results_" + str(func_name), aca_perc_results)
        np.save("results/solrad1_perc_results_" + str(func_name), solrad1_perc_results)
        np.save("results/solrad2_perc_results_" + str(func_name), solrad2_perc_results)
    return


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
    cp = make_cluster_problem(data_name, func_name)

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
        approx, cp = test_sprl(data_name, 40, sample_factor=sample_factor, copy_cp=cp)
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


def calc_median(array2d):
    size = len(array2d)
    medians = np.zeros(size)
    for i in range(size):
        medians[i] = np.median(array2d[i])
    return medians


def calc_max(array2d):
    size = len(array2d)
    maxs = np.zeros(size)
    for i in range(size):
        maxs[i] = np.max(array2d[i])
    return maxs


def get_number_of_tests(start_rank, max_rank, step):
    i = 0
    current = start_rank
    while current < max_rank:
        i += 1
        current += step
    return i


def test_aca(data_name, max_rank, epsilon=None, debug=False, copy_cp=None):
    if copy_cp is None:
        cp = make_cluster_problem(data_name, "msm", debug=debug)
    else:
        cp = copy_cp.make_non_sampled_copy()
    approx = aca.aca_symm(cp, tolerance=epsilon, max_rank=max_rank, debug=debug)
    return approx, cp


def test_solrad(data_name, rank, epsilon=None, debug=False, copy_cp=None):
    if copy_cp is None:
        cp = make_cluster_problem(data_name, "msm", debug=debug)
    else:
        cp = copy_cp.make_non_sampled_copy()
    approx = solrad.solrad(cp, rank=rank, epsilon=epsilon, debug=debug)
    return approx, cp


def test_sprl(data_name, rank, sample_factor=None, debug=False, copy_cp=None):
    if copy_cp is None:
        cp = make_cluster_problem(data_name, "msm", debug=debug)
    else:
        cp = copy_cp.make_non_sampled_copy()
    distances_with_zero = sprl.compute_distances_with_zero(cp.series, cp.compare)
    temp = cp.solved_matrix
    print("Converting to similarity")
    cp.solved_matrix = sprl.convert_distance_matrix_to_similarity_matrix(cp.solved_matrix, distances_with_zero)
    approx = sprl.convert_similarity_matrix_to_distance_matrix(sprl.spiral(cp, rank, sample_factor=sample_factor), distances_with_zero)
    cp.solved_matrix = temp
    return approx, cp


def make_cluster_problem(data_name, df_name, debug=False):
    data = dl.read_train_and_test_data(data_name, debug=debug)
    if df_name == "dtw":
        matrix = dl.compute_distance_matrix_dtw(data, data_name)
        cp = cluster_problem.ClusterProblem(data, dtw_fast, solved_matrix=matrix)
    elif df_name == "ed":
        matrix = dl.compute_distance_matrix_ed(data, data_name)
        cp = cluster_problem.ClusterProblem(data, ed_fast, solved_matrix=matrix)
    else:
        matrix = dl.compute_distance_matrix_msm(data, data_name)
        cp = cluster_problem.ClusterProblem(data, msm_fast, solved_matrix=matrix)
    return cp


if __name__ == "__main__":
    name = "ECG5000"
    func_name = "msm"
    max_rank = 10

    #run_increasing_rank_test(name, func_name, max_rank=90, start_rank=10, step=20, test_size=1)
    #run_efficiency_test(name, func_name, max_rank=50, start_rank=1, step=2, test_size=1)
    #run_all_datasets_better_than_accuracy_test(func_name, [0.1, 0.05, 0.02, 0.01], start_index=0)
    #increasing_sample_percentage_sprl(name, func_name)
    show_results_efficiency_test(name)

    #approx1, approx2 = run_compare_test(name, max_rank, accuracy, debug=True)
    #approx = test_solrad(name, accuracy, start_rank=fixed_rank, debug=True)

    #run_best_rank_error(name, max_rank=200, step=5, start_rank=5)
    #run_singular_values_plot(name, 200)

    #run_increasing_input_size(data_name=name, debug=False)

    # amount = 10
    # adapt_errors = np.zeros(amount)
    # fixed_errors = np.zeros(amount)
    # factors = np.zeros(amount)
    # for i in range(amount):
    #     factor = (i*0.2 + 1.0)
    #     factors[i] = factor
    #     accuracy = 1.0 / factor
    #     adapt_errors[i], fixed_errors[i] = run_compare_solrad_algos(name, epsilon=accuracy)
    #
    # multiple_scatter_plot((adapt_errors, fixed_errors), (factors, factors),
    #                         ('r', 'b'), ('ADAPTIVE', 'FIXED'),
    #                         'Sample factor regression', 'Relative Error', 'Error based on regression sample factor')