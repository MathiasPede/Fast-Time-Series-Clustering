from ftsc.data_loader import load_timeseries_from_multiple_tsvs
from ftsc.cluster_problem import ClusterProblem
import os
import numpy as np

folder = "Data/"
matrix_data_folder = "Matrix_data/"
sing_vals_folder = "Singular_values/"


def create_cluster_problem(data_name, func_name):
    train_path = folder + data_name + "/" + data_name + "_TRAIN.tsv"
    test_path = folder + data_name + "/" + data_name + "_TEST.tsv"
    labels, series = load_timeseries_from_multiple_tsvs(train_path, test_path)

    solved_matrix = load_matrix(data_name, func_name)

    cp = ClusterProblem(series, func_name, solved_matrix=solved_matrix)
    return cp


def get_labels(data_name):
    train_path = folder + data_name + "/" + data_name + "_TRAIN.tsv"
    test_path = folder + data_name + "/" + data_name + "_TEST.tsv"
    labels, _ = load_timeseries_from_multiple_tsvs(train_path, test_path)
    return labels


def get_amount_of_classes(labels):
    return len(np.unique(labels))


def load_matrix(data_name, func_name):
    file_path = matrix_data_folder + data_name + "_" + func_name + ".npy"
    if os.path.isfile(file_path):
        solved_matrix = np.load(file_path)
    else:
        solved_matrix = None
    return solved_matrix


def take_submatrix_matrix(matrix, indices):
    submatrix = matrix[np.ix_(indices, indices)]
    return submatrix


def load_singular_values(data_name, func_name):
    file_path = sing_vals_folder + "SV_" + data_name + "_" + func_name + ".npy"
    if os.path.isfile(file_path):
        sing_vals = np.load(file_path)
    else:
        sing_vals = None
    return sing_vals


def get_all_test_dataset_names():
    lst = [x[0] for x in os.walk("Data")][1:]
    drop_directory = [x[5:] for x in lst]
    return drop_directory
