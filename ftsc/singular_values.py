import numpy as np
from math import sqrt
from data_loader import load_array


def calculate_best_relative_error_rank(data_name, func_name, rank):
    sing_vals = get_singular_values(data_name, func_name)
    if rank > len(sing_vals):
        return 0.0
    size = len(sing_vals)
    squared = np.square(sing_vals)
    absolute_error = sqrt(np.sum(squared[::-1][0:size-rank]))
    norm = sqrt(np.sum(squared))
    return absolute_error / norm


def calculate_best_relative_error_for_all_ranks(sing_vals):
    squared = np.square(sing_vals)
    norm = sqrt(np.sum(squared))
    abs_error = np.sqrt(np.cumsum(squared[::-1])[::-1])
    return abs_error / norm


def get_first_rank_relative_error_below(data_name, func_name, rel_error):
    sing_vals = get_singular_values(data_name, func_name)
    rel_errors = calculate_best_relative_error_for_all_ranks(sing_vals)
    index = np.where(rel_errors < rel_error)[0][0]
    return index + 1


def get_singular_values(data_name, func_name):
    folder = "Singular_values/"
    file_name = folder + "SV_" + data_name + "_" + func_name
    try:
        return np.load(file_name + ".npy")
    except IOError:
        matrix = load_array(data_name, func_name)
        sing_vals = np.linalg.svd(matrix)[1]
        np.save(file_name, sing_vals)
        return sing_vals


if __name__ == '__main__':
    import data_loader as dl
    from plotter import scatter_plot
    datasets = dl.get_all_dataset_names()
    size = len(datasets)

    func_name = "dtw"
    rank = 20
    errors = np.zeros(size)

    for i in range(size):
        name = datasets[i]
        errors[i] = calculate_best_relative_error_rank(name, func_name, rank)

    average = np.average(errors)
    max_error = np.max(errors)
    std = np.std(errors)
    print("AVERAGE: " + str(average))
    print("MAX: " + str(max_error))
    print("STD: " + str(std))
    sizes = np.load("results/dataset_sizes.npy")
    scatter_plot(errors, sizes)

    np.save("results/best_errors_rank_" + str(rank), errors)