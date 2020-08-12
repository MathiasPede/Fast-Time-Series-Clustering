"""
TRIANGLE FIXING TO FIND THE NEAREST MATRIX THAT HAS THE TRIANGLE INEQUALITY PROPERTY
"""
import numpy as np
import random as rnd
from helper import distances_array_to_matrix_with_diagonal, distances_matrix_to_array_with_diagonal, _print_library_missing

triangle_fixing_c = None
try:
    import triangle_fixing_c
except ImportError:
    # logger.info('C library not available')
    triangle_fixing_c = None


def triangle_fixing_fast(matrix, violation_margin=0.01):
    if triangle_fixing_c is None:
        _print_library_missing()
        return None
    if violation_margin is None:
        violation_margin=0.01

    flat = distances_matrix_to_array_with_diagonal(matrix)

    output = triangle_fixing_c.triangle_fixing_nogil(flat, len(matrix), violation_margin)

    result = distances_array_to_matrix_with_diagonal(output, len(matrix))

    return result


def get_number_of_violations_fast(matrix, violation_margin=0.01):
    if triangle_fixing_c is None:
        _print_library_missing()
        return None
    if violation_margin is None:
        violation_margin=0.01

    flat = distances_matrix_to_array_with_diagonal(matrix)

    number = triangle_fixing_c.get_number_of_violations_nogil(flat, len(matrix), violation_margin)

    return number


class SymmetricMatrix:
    def __init__(self, size, dtype=float):
        self.size = size
        self.dtype = dtype
        self.array = np.zeros(((size+1)*size) // 2, dtype=self.dtype)

    def set(self, i, j, value):
        if j > i:
            i, j = j, i
        index = (((2*self.size - i + 1) * i) // 2) + (j-i)
        self.array[index] = value

    def get(self, i, j):
        if j > i:
            i, j = j, i
        index = (((2*self.size - i + 1) * i) // 2) + (j-i)
        return self.array[index]

    def to_matrix(self):
        result = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                result[i, j] = self.get(i, j)
        return result


class UpperTriangleOfArrays:
    def __init__(self, size, array_size):
        self.size = size
        self.array_size = array_size
        flat_size = ((size+1)*size) // 2
        self.array = np.zeros((flat_size, array_size))

    def set(self, i, j, k, value):
        assert i <= j
        index = (((2*self.size - i + 1) * i) // 2) + (j-i)
        self.array[index, k] = value

    def get(self, i, j, k):
        assert i <= j
        index = (((2*self.size - i + 1) * i) // 2) + (j-i)
        return self.array[index, k]


def triangle_fixing(matrix, violation_margin=0.01, tolerance=0.01):
    # Initialize error and correction matrix
    size = len(matrix)
    errors = SymmetricMatrix(size)
    correction_terms = UpperTriangleOfArrays(size, size)

    # Loop parameter
    # diff = 1 + tolerance
    # start = 0
    violations = size

    while violations > tolerance*size:
        violations = 0
        for i in range(size):
            for j in range(i+1, size):
                for k in range(size):
                    if i == k or j == k:
                        continue

                    b = matrix[k, i] + matrix[j, k] - matrix[i, j]

                    eij = errors.get(i, j)
                    ejk = errors.get(j, k)
                    eki = errors.get(k, i)

                    mu = (1.0/3.0) * (eij - ejk - eki - b)

                    if mu > 0:
                        if mu > violation_margin:
                            violations += 1
                        zijk = correction_terms.get(i, j, k)
                        theta = min(-mu, zijk)

                        print("Theta: " + str(theta))

                        errors.set(i, j, eij + theta)
                        errors.set(j, k, ejk - theta)
                        errors.set(k, i, eki - theta)

                        # print("Violation: " + str(i) + " " + str(j) + " " + str(k) + " " + str(b))
                        # print("Errors: " + str(eij) + " " + str(ejk) + " " + str(eki))
                        # print(str(mu))
                        # print(str(zijk))
                        # print(str(theta))

                        correction_terms.set(i, j, k, zijk - theta)

        print("Number of triangle violations: " + str(violations))

    error_matrix = errors.to_matrix()
    return np.add(matrix, error_matrix)


if __name__ == '__main__':
    import data_loader as dl

    datasets = dl.get_all_dataset_names()
    size = len(datasets)
    relative_errors = np.zeros(size)
    indices = [8,9,10,13,17,18,19,20,21,23,24,26,27,28,29,33,37,38,42,44,45,46,47,48,49,52,53,60,64,65,66,67,68,73,74,76,78,81]
    # for i in range(size):
    #     name = datasets[i]
    #     data = dl.read_train_and_test_data(name, debug=False)
    #     if len(data) > 1500:
    #         print("Too large: " + name)
    #         continue
    #     matrix = dl.compute_distance_matrix_dtw(data, name)
    #     frob_norm = np.linalg.norm(matrix)
    #
    #     fixed = triangle_fixing_fast(matrix, violation_margin=0.01)
    #     diff = np.subtract(matrix, fixed)
    #     diff_norm = np.linalg.norm(diff)
    #
    #     relative_errors[i] = diff_norm / frob_norm
    #
    relative_errors = np.take(np.load("results/triangle_relative_errors.npy"), indices)
    #
    # matrix = dl.load_array("CinCECGTorso", "dtw")
    # #fixed1 = triangle_fixing(matrix, violation_margin=0.01, tolerance=0.01)
    # fixed2 = triangle_fixing_fast(matrix, violation_margin=0.01, tolerance=0.01)
    #
    # #diff1 = np.subtract(matrix, fixed1)
    # diff2 = np.subtract(matrix, fixed2)
    #
    # #norm1 = np.linalg.norm(diff1)
    # norm2 = np.linalg.norm(diff2)
