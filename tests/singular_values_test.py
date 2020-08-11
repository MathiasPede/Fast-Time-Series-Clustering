from data_loader import load_array, compute_distance_matrix, get_all_dataset_names, save_array
from plotter import scatter_plot, multiple_scatter_plot
import numpy as np
from math import sqrt, floor
from singular_values import get_singular_values, calculate_best_relative_error_for_all_ranks


def run_best_error_plot(data_name, max=None):
    sing_vals_dtw = get_singular_values(data_name, "dtw")
    sing_vals_msm = get_singular_values(data_name, "msm")
    sing_vals_ed = get_singular_values(data_name, "ed")

    if max is None or max > len(sing_vals_dtw):
        max = floor(0.99 * len(sing_vals_dtw))

    xas = np.arange(0, max)
    error_dtw = calculate_best_relative_error_for_all_ranks(sing_vals_dtw)[0:max]
    error_msm = calculate_best_relative_error_for_all_ranks(sing_vals_msm)[0:max]
    error_ed = calculate_best_relative_error_for_all_ranks(sing_vals_ed)[0:max]

    multiple_scatter_plot((error_dtw, error_msm, error_ed), (xas, xas, xas), colors=["blue", "red", "green"],
                          labels=["DTW", "MSM", "ED"], xname="Rang benadering", yname="Relatieve fout",
                          title="Beste relatieve fout voor benaderingsrang " + data_name)


def run_singular_values_plot(data_name, func_name):
    sing_vals= get_singular_values(data_name, "dtw")

    max = floor(0.99 * len(sing_vals))
    xas = np.arange(0, max)
    scatter_plot(sing_vals[0:max], xas, yname="Singuliere waarden", xname="Index", title="Singuliere waarden van" + func_name + " matrix " + data_name)


def get_least_rank_for_error(data_name, func_name, error):
    sing_vals = get_singular_values(data_name, func_name)
    errors = calculate_best_relative_error_for_all_ranks(sing_vals)
    rank = np.where(errors <= error)
    try:
        return rank[0][0]
    except IndexError:
        return len(sing_vals)


def get_percentage_least_rank(data_name, func_name, error):
    sing_vals = get_singular_values(data_name, func_name)
    errors = calculate_best_relative_error_for_all_ranks(sing_vals)
    rank = np.where(errors <= error)
    try:
        return float(rank[0][0]) / float(len(sing_vals))
    except IndexError:
        return 1


if __name__ == '__main__':
    datasets = get_all_dataset_names()
    dtw_ranks = []
    msm_ranks = []
    max_error = 0.005

    for name in datasets:
        name= "ECG5000"
        run_singular_values_plot(name, "dtw")
        run_best_error_plot(name)

        rank_dtw = get_percentage_least_rank(name, "dtw", max_error)
        dtw_ranks.append(rank_dtw)
        rank_msm = get_percentage_least_rank(name, "msm", max_error)
        msm_ranks.append(rank_msm)

    difference = np.subtract(msm_ranks, dtw_ranks)

    dtw_average = sum(dtw_ranks) / len(dtw_ranks)
    msm_average = sum(msm_ranks) / len(msm_ranks)

    difference_average = sum(difference) / len(difference)

    dtw_better = sum(x > 0 for x in difference)
    dtw_better_percentage = dtw_better / float(len(difference))
    same = sum(x == 0 for x in difference)
    same_percentage = same / float(len(difference))
    msm_better = sum(x < 0 for x in difference)
    msm_percentage = msm_better / float(len(difference))

    dtw_std = sqrt(np.var(dtw_ranks))
    msm_std = sqrt(np.var(msm_ranks))

    #difference_std = sqrt(np.var(difference))

    print("AVERAGE: " + str(dtw_average) + " vs " + str(msm_average))
    print("STD: " + str(dtw_std) + " vs " + str(msm_std))
    print("DTW : " + str(dtw_better) + " EQUAL: " + str(same) + " MSM : " + str(msm_better))
