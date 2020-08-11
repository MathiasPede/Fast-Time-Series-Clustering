"""
    Class ClusterProblem

    Used for storing the distance/similarity matrix. Starts with uninitialized elements

    Can be sampled
"""

import numpy as np
import math
import best_low_rank_approximation as blra


class ClusterProblem:
    def __init__(self, series, compare, start_index=1, solved_matrix=None):
        """
        Creates a cluster problem objects, which contains the data objects. A distance matrix is created with initially
        NaN values and entries can be computed based on index using the 'compare' function
        :param series: The data objects (array of time series, usually with first index the class number)
        :param compare: Distance function
        :param start_index: Index of each object array where the time series start (usually 1, after the class nr)
        :param solved_matrix: Potentially add an already solved matrix to speed up
        """
        self.series = series
        self.start_index = start_index
        self.compare = compare
        self.size = len(series)
        self.matrix = np.empty((self.size, self.size), dtype=np.float_)
        self.matrix[:] = np.NaN
        self.svd = None
        self.amount_sampled = 0
        self.omega = None
        self.solved_matrix = solved_matrix

    def cp_size(self):
        return self.size

    def get_time_serie(self, x):
        return self.series[x][self.start_index:]

    def read(self, x, y):
        return self.matrix[x][y]

    def write(self, x, y, val):
        self.matrix[x][y] = val
        self.matrix[y][x] = val

    def sample(self, x, y):
        # if the element has not yet been calculated, calculate
        value = self.read(x, y)
        if math.isnan(value):
            if self.solved_matrix is not None:
                value = self.solved_matrix[x][y]
            else:
                value = self.compare(self.get_time_serie(x), self.get_time_serie(y))
            self.write(x, y, value)
        return value

    def sample_all(self, indices):
        size = len(indices)
        values = np.zeros(size)
        for i in range(size):
            values[i] = self.sample(indices[i][0], indices[i][1])
        return values

    def sample_row(self, row):
        if self.solved_matrix is not None:
            self.matrix[row, :] = self.solved_matrix[row, :]
        else:
            for i in range(self.cp_size()):
                self.sample(row, i)
        return self.matrix[row, :]

    def sample_col(self, col):
        if self.solved_matrix is not None:
            self.matrix[:, col] = self.solved_matrix[:, col]
        else:
            for i in range(self.cp_size()):
                self.sample(i, col)
        return self.matrix[:, col]

    def sample_full_matrix(self):
        if self.solved_matrix is None:
            for i in range(self.cp_size()):
                self.sample_row(i)
        else:
            self.matrix = self.solved_matrix
        return self.matrix

    def sample_diagonal(self):
        diagonal = np.zeros(self.cp_size())
        for i in range(self.cp_size()):
            diagonal[i] = self.sample(i,i)
        return diagonal

    def compute_omega(self):
        """
        Compute the omega list from SPRL algorithm containing the indices per row that have been sampled
        """
        if self.omega is None:
            self.omega = np.empty(self.cp_size(), dtype=np.ndarray)
            for x in range(self.cp_size()):
                non_nan_indices = np.argwhere(~np.isnan(self.matrix[x, :]))
                # observed_indices = []
                # for y in range(self.cp_size()):
                #     if np.isnan(self.read(x,y)):
                #         continue
                #     else:
                #         observed_indices.append(y)
                self.omega[x] = np.array(non_nan_indices)
        return self.omega

    def get_omega_row(self, row):
        return self.omega[row]

    def get_amount_of_sampled_values(self):
        # count non nan numbers on diagonal
        diag_number = 0
        for i in range(self.cp_size()):
            if not np.isnan(self.matrix[i,i]):
                diag_number += 1
        # count all non nan numbers
        non_nan_numbers = np.count_nonzero(~np.isnan(self.matrix))
        return (non_nan_numbers - diag_number) / 2.0 + diag_number

    def percentage_sampled(self):
        return self.get_amount_of_sampled_values() / self.get_max_sample_amount()

    def get_max_sample_amount(self):
        return self.size * (self.size + 1) / 2.0

    def get_svd(self):
        if not self.svd:
            self.svd = np.linalg.svd(self.sample_full_matrix())
        return self.svd

    def get_singular_values(self):
        u, s, v = self.get_svd()
        return s

    def get_best_approx_for_rank(self, k):
        return blra.low_rank_approx(SVD=self.get_svd(), r=k)

    def get_frobenius_norm(self):
        if self.solved_matrix is not None:
            result = np.linalg.norm(self.solved_matrix)
        else:
            result = np.linalg.norm(self.sample_full_matrix())
        return result

    def get_relative_error(self, approx):
        if self.solved_matrix is not None:
            error = np.subtract(self.solved_matrix, approx)
        else:
            error = np.subtract(self.sample_full_matrix(), approx)
        norm_error = np.linalg.norm(error)
        return norm_error / self.get_frobenius_norm()

    def get_best_possible_error(self, rank):
        return self.get_relative_error(self.get_best_approx_for_rank(rank))

    def get_cluster_labels(self):
        result = np.array(self.cp_size(), dtype=int)
        for i in range(self.cp_size()):
            result[i] = self.series[i][0]
        return result

    def make_non_sampled_copy(self):
        return ClusterProblem(self.series, self.compare, start_index=self.start_index, solved_matrix=self.solved_matrix)

    def reset_matrix(self):
        self.matrix[:] = np.NaN
