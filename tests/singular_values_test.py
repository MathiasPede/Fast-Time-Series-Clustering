import numpy as np

from math import floor
from ftsc.singular_values import calculate_best_relative_error_for_all_ranks, calculate_best_relative_error_rank
from tests.plotting_utils import scatter_plot, multiple_scatter_plot
from tests.tests_utils import get_singular_values, get_all_test_dataset_names


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
                          labels=["DTW", "MSM", "ED"], xname="Rang $k$ van de benadering",
                          yname="Relatieve fout $\eta$",
                          title="Beste relatieve fout voor benaderingsrang " + data_name, marker='o')


def run_singular_values_plot(data_name, func_name, max=None):
    sing_vals = get_singular_values(data_name, "dtw")

    if max is None:
        max = floor(0.99 * len(sing_vals))
    xas = np.arange(0, max)
    scatter_plot(sing_vals[0:max], xas, yname="Singuliere waarden $\sigma_i$", xname="Index $i$",
                 title="Singuliere waarden van " + func_name + " matrix " + data_name, marker='o')


if __name__ == '__main__':
    func_name = "ed"

    # Singular values and errors for 1 dataset
    name = "Crop"
    run_singular_values_plot(name, func_name)
    run_best_error_plot(name, max=None)

    # Compare all dataset error for certain ranks
    all_datasets = get_all_test_dataset_names()
    size = len(all_datasets)
    ranks = [20, 50, 100, 200]

    errors = np.zeros(shape=(size, len(ranks)))

    for i in range(size):
        name = all_datasets[i]
        sing_vals = get_singular_values(name, func_name)
        for j in range(len(ranks)):
            errors[i, j] = calculate_best_relative_error_rank(sing_vals, ranks[j])

    averages = np.average(errors, axis=0)
    stds = np.std(errors, axis=0)
    max = np.max(errors, axis=0)
