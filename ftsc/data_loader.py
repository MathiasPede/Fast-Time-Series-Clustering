import numpy as np
import random as rnd
import os.path
from os import walk
from dtaidistance.dtw import distance_matrix_fast
from msm import msm_matrix_fast, msm_fast
from ed import ed_matrix_fast, ed_fast
import distance_functions as df

folder = "Data/"

matrix_folder = "Matrix_data/"


def save_array(arr, data_name, distance_func_name):
    file_name = matrix_folder + data_name + "_" + distance_func_name + ".npy"
    np.save(file_name, arr)
    print("Saved array to file " + file_name)


def load_array(data_name, distance_func_name):
    file_name = matrix_folder + data_name + "_" + distance_func_name + ".npy"
    print("Loading array from file " + file_name)
    arr = np.load(file_name)
    return arr


def has_saved_array(data_name, distance_func_name):
    file_name = matrix_folder + data_name + "_" + distance_func_name + ".npy"
    return os.path.exists(file_name)


def read_test_data(name, debug=False):
    path = folder + name + "/" + name + "_TEST.tsv"
    return read_data(path, debug=debug)


def read_train_data(name, debug=False):
    path = folder + name + "/" + name + "_TRAIN.tsv"
    return read_data(path, debug=debug)


def read_train_and_test_data(name, debug=False):
    train_data = read_train_data(name, debug=debug)
    train_size = len(train_data)
    test_data = read_test_data(name, debug=debug)
    test_size = len(test_data)
    data = np.concatenate((train_data, test_data), axis=0)
    return data


def read_data(path, debug=False):
    if debug:
        print("Loading data from: " + path)
    data = np.genfromtxt(path, delimiter='\t')
    if debug:
        print("Loaded " + str(len(data)) + " data entries")
    return data


def sample_data(data, size, seed=None, debug=False):
    if seed:
        rnd.seed(seed)
    if debug:
        print("Sub-sampling the data with seed " + str(seed) + " and size " + str(size))
    indices = rnd.sample(range(len(data)), size)
    return data[tuple(indices)]


def compute_distance_matrix_dtw(data, data_name):
    if has_saved_array(data_name, "dtw"):
        return load_array(data_name, "dtw")
    data_without_class = data[:, 1:]
    matrix = distance_matrix_fast(data_without_class)
    for i in range(len(data)):
        matrix[i, i] = df.dtw(data[i][1:], data[i][1:])
    mirror_top_triangle_matrix(matrix)
    save_array(matrix, data_name, "dtw")
    return matrix


def mirror_top_triangle_matrix(matrix):
    # set every column equal to rows (IN ORDER)
    for i in range(len(matrix)):
        matrix[:, i] = matrix[i, :]


def compute_distance_matrix_ed(data, data_name):
    if has_saved_array(data_name, "ed"):
        return load_array(data_name, "ed")
    data_without_class = data[:, 1:]
    matrix = ed_matrix_fast(data_without_class)
    for i in range(len(data)):
        matrix[i, i] = df.dtw(data[i][1:], data[i][1:])
    mirror_top_triangle_matrix(matrix)
    save_array(matrix, data_name, "ed")
    return matrix


def compute_distance_matrix_msm(data, data_name):
    if has_saved_array(data_name, "msm"):
        return load_array(data_name, "msm")
    data_without_class = data[:, 1:]
    matrix = msm_matrix_fast(data_without_class)
    for i in range(len(data)):
        matrix[i, i] = msm_fast(data[i][1:], data[i][1:])
    mirror_top_triangle_matrix(matrix)
    save_array(matrix, data_name, "msm")
    return matrix


def compute_distance_matrix(data, data_name, func):
    size = len(data)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            serie1 = data[i][1:]
            serie2 = data[j][1:]
            value = func(serie1, serie2)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def take_submatrix_matrix(matrix, indices):
    submatrix = matrix[np.ix_(indices, indices)]
    return submatrix


def get_cluster_labels(data):
    result = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        result[i] = int(data[i][0])
    return result


def get_amount_of_classes(data):
    return len(np.unique(get_cluster_labels(data)))


def get_all_dataset_names():
    lst = [x[0] for x in walk("Data")][1:]
    drop_directory = [x[5:] for x in lst]
    return drop_directory


def get_singular_values(data_name, func_name):
    folder = "Singular_values/"
    file_name = folder + "SV_" + data_name + "_" + func_name
    try:
        return np.load(file_name + ".npy")
    except IOError:
        matrix = load_array(data_name, func_name)
        sing_vals_dtw = np.linalg.svd(matrix)[1]
        np.save(file_name, sing_vals_dtw)
        return sing_vals_dtw


if __name__ == '__main__':
    from sprl import convert_distance_matrix_to_similarity_matrix, compute_distances_with_zero
    from dtaidistance import dtw

    all_dataset_names = get_all_dataset_names()
    sizes = np.zeros(len(all_dataset_names), dtype=int)

    for i in range(len(all_dataset_names)):
        name = all_dataset_names[i]
        print("Computing dataset: " + name)
        data = read_train_and_test_data(name)
        print("Size:" + str(len(data)))
        sizes[i] = len(data)

    np.save("results/dataset_sizes", sizes)


