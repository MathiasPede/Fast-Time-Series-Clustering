import numpy as np

from math import floor
from ftsc.singular_values import calculate_best_relative_error_for_all_ranks, calculate_best_relative_error_rank
from tests.plotting_utils import scatter_plot, multiple_scatter_plot
from tests.tests_utils import load_singular_values, create_cluster_problem, get_all_test_dataset_names


def get_singular_values(data_name, func_name):
    # If stored in memory
    sing_vals = load_singular_values(data_name, func_name)
    if sing_vals is None:
        cp = create_cluster_problem(data_name, func_name)
        sing_vals = cp.get_singular_values()
    return sing_vals


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


func_name = "dtw"

# Singular values and errors for 1 dataset
name = "ECG5000"
run_singular_values_plot(name, func_name)
run_best_error_plot(name)


# Compare all dataset error for certain ranks
all_datasets = get_all_test_dataset_names()
size = len(all_datasets)
ranks = [5, 10, 20, 50, 100]

errors = np.zeros(shape=(size, len(ranks)))

for i in range(size):
    name = all_datasets[i]
    sing_vals = get_singular_values(name, func_name)
    for j in range(len(ranks)):
        errors[i,j] = calculate_best_relative_error_rank(sing_vals, ranks[j])

averages = np.average(errors, axis=0)
stds = np.std(errors, axis=0)
max = np.max(errors, axis=0)