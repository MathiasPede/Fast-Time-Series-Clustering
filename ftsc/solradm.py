"""
    Sample Optimal Low Rank Approximation of Distance Matrices
"""
from scipy import linalg
import numpy as np
import random as rnd
import ftsc.cluster_problem as cp
from math import sqrt, floor
import warnings

warnings.filterwarnings('ignore')


def compute_probabilities(clust_prob, rows_amount=1):
    """
    Calculate the probabilities for the sampling algorithm by approximating the norms of the
    rows using a reference element of the matrix.
    :param cp: ClusterProblem (contains the matrix)
    :return: numepy array with probabilities (normalized already)
    """
    if rows_amount is None:
        rows_amount = 1

    # Initialize list of probabilities
    size = clust_prob.cp_size()

    best_norm = np.infty
    best_row = None
    for i in range(rows_amount):
        # Choose a random time series as reference
        ref = rnd.randint(0, size - 1)

        ref_row = clust_prob.sample_row(ref)
        norm = np.linalg.norm(ref_row)
        if norm < best_norm:
            best_norm = norm
            best_row = ref_row

    squared = np.square(best_row)
    average_distance_to_ref = np.average(squared)
    arr = squared + average_distance_to_ref
    result = arr / np.sum(arr)

    return result


def compute_u_with_svd(clustp, probs, k=None, epsilon=0.05):
    """
    Create a matrix U containing approximations of the top k left singular vectors
    :param delta: Likelihood of approximation being better than the epsilon given
    :param clustp: ClusterProblem
    :param probs: Sampling probabilities (length n)
    :param k: rank of approximation
    :param epsilon: Accuracy
    :return: n x k matrix U
    """
    if not k:
        k = 20
    if not epsilon:
        epsilon = 0.05

    s = min(floor(10 * k / epsilon), clustp.cp_size())
    print("Sampling amount to compute column space: " + str(s))

    reduced = np.zeros(shape=(clustp.cp_size(), s))

    col_indices = []
    for i in range(s):
        number = np.random.choice(np.arange(0, clustp.cp_size()), p=probs)
        while number in col_indices:
            number = np.random.choice(np.arange(0, clustp.cp_size()), p=probs)
        col_indices.append(number)

        reduced[:, i] = clustp.sample_col(number) / sqrt(s * probs[number])

    left, sigmas, right = linalg.svd(reduced)

    # u = np.zeros(shape=(clustp.cp_size(), k))
    # for i in range(k):
    #     u[:, i] = np.matmul(reduced, right[i]) * (1.0 / sigmas[i]*sigmas[i])
    #     u[:, i] = u[:, i] / np.linalg.norm(u[:, i])
    u = left[:, 0:k]

    return u


def compute_u_subsampling(clustp, probs, k=None, epsilon=0.05, debug=False):
    """
    Create a matrix U containing approximations of the top k left singular vectors
    :param delta: Likelihood of approximation being better than the epsilon given
    :param clustp: ClusterProblem
    :param probs: Sampling probabilities (length n)
    :param k: rank of approximation
    :param epsilon: Accuracy
    :return: n x k matrix U
    """
    if not k:
        k = 20
    if not epsilon:
        epsilon = 0.05

    s = min(floor(10 * k / (epsilon)), clustp.cp_size())
    if debug:
        print("Sampling amount to compute column space: " + str(s))

    # Sample the columns of original matrix
    reduced = np.zeros(shape=(clustp.cp_size(), s))
    col_indices = []
    for i in range(s):
        number = np.random.choice(np.arange(0, clustp.cp_size()), p=probs)
        while number in col_indices:
            number = np.random.choice(np.arange(0, clustp.cp_size()), p=probs)
        col_indices.append(number)

        reduced[:, i] = clustp.sample_col(number) / sqrt(s * probs[number])

    chosen_col_probabilities = probs[col_indices]
    squared = np.square(reduced)
    sum_of_cols = np.sum(squared, axis=0)
    normalized_squared = squared / sum_of_cols
    adjusted = normalized_squared / chosen_col_probabilities
    row_probs = np.sum(adjusted, axis=1)
    row_probs_normalized = row_probs / np.sum(row_probs)

    # Sample the rows of the reduced matrix
    w_matrix = np.zeros(shape=(s, s))
    row_indices = []
    for i in range(s):
        number = np.random.choice(np.arange(0, clustp.cp_size()), p=row_probs_normalized)
        while number in row_indices:
            number = np.random.choice(np.arange(0, clustp.cp_size()), p=row_probs_normalized)
        row_indices.append(number)

        w_matrix[i] = reduced[number] / sqrt(s * row_probs_normalized[number])

    left, sigmas, right = np.linalg.svd(w_matrix)

    u = np.zeros(shape=(clustp.cp_size(), k))

    for i in range(k):
        u[:, i] = np.matmul(reduced, right[i])
        u[:, i] = u[:, i] / np.linalg.norm(u[:, i])

    return u


def well_balanced_linear_regression(cp, X, epsilon=0.05, debug=False):
    n = X.shape[0]
    k = X.shape[1]

    indices, weights = active_sampling(cp, X, epsilon=epsilon)
    indices_size = len(indices)
    sqrt_weights = np.sqrt(weights)

    if debug:
        print("Sampling amount to solve regression: " + str(indices_size))

    A = np.take(X, indices, axis=0)
    A_weighted = A * sqrt_weights[:, None]

    b_weighted = np.zeros((indices_size, n))

    for i in range(len(indices)):
        index = indices[i]
        weight = sqrt_weights[i]
        b_weighted[i] = weight * cp.sample_row(index)

    v_approx = np.zeros((k, n))
    for i in range(n):
        v_approx[:, i] = np.linalg.lstsq(A_weighted, b_weighted[:, i])[0]

    return v_approx


def active_sampling(cp, X, epsilon=0.05):
    """
    Uses well balanced sampling method to pick rows from X to sample
    :param cp: Cluster problem
    :param X: Features of all the objects of the cluster problem
    :param epsilon: Maximal additive error of regression
    :return: (indices, weights) of the sampled objects
    """
    n = X.shape[0]
    k = X.shape[1]
    if epsilon is None:
        epsilon = 0.05

    basis = find_orthonormal_basis(k, X)
    # basis_test = find_orthonormal_basis(k, X, given=basis)
    # basis = make_normal_indentity(k, X)

    possible_indices = np.arange(0, n)
    indices = []
    weights = []

    gamma = sqrt(epsilon) / 3.0
    mid = (4.0 * k / gamma) / ((1.0 / (1.0 - gamma)) - (1.0 / (1.0 + gamma)))
    j = 0
    B = np.zeros((k, k))
    l = -2.0 * k / gamma
    u = 2.0 * k / gamma
    D_new = np.zeros(n)
    threshold = 8.0 * k / gamma

    alpha_coefs = []
    v_matrix = np.matmul(X, basis)

    while u - l < threshold or j == n:
        upper = u * np.identity(k) - B
        lower = B - l * np.identity(k)
        inverse_upper = linalg.inv(upper)
        inverse_lower = linalg.inv(lower)

        phi = np.trace(inverse_upper) + np.trace(inverse_lower)
        alpha_coefs.append(gamma / (phi * mid))

        factors1_temp = np.matmul(inverse_upper, np.transpose(v_matrix))
        factors1 = np.einsum("ij,ij->i", v_matrix, np.transpose(factors1_temp))
        factors2_temp = np.matmul(inverse_lower, np.transpose(v_matrix))
        factors2 = np.einsum("ij,ij->i", v_matrix, np.transpose(factors2_temp))

        D_new = np.add(factors1, factors2)
        norm = np.sum(D_new)

        sampled_index = np.random.choice(possible_indices, p=(D_new / norm))
        indices.append(sampled_index)
        sample = X[sampled_index]

        scale = (gamma) / (D_new[sampled_index])
        weights.append(scale / mid)

        B = B + scale * np.outer(v_matrix[sampled_index], v_matrix[sampled_index])
        u += (gamma / (phi * (1 - gamma)))
        l += (gamma / (phi * (1 + gamma)))
        j += 1

    return indices, np.array(weights)


def fix_diagonal(approximation):
    """
    Set all the diagonal elements to 0.0 (distance matrix)
    :param approximation: Matrix
    :return: Matrix with only 0.0 on diagonal
    """
    size = approximation.shape[0]
    for i in range(size):
        approximation[i, i] = 0.0
    return approximation


def solrad(cp, rank, epsilon=0.05, debug=False):
    if epsilon is None:
        epsilon = 0.05

    # Three steps of SOLRADM algorithm
    probs = compute_probabilities(cp, rows_amount=rank)
    # u = compute_u_with_svd(cp, probs, k=rank, epsilon=epsilon)
    u = compute_u_subsampling(cp, probs, k=rank, epsilon=epsilon, debug=debug)
    x = well_balanced_linear_regression(cp, u, epsilon=epsilon, debug=debug)

    # Compute the low rank approximation
    approx = np.matmul(u, x)
    np.fill_diagonal(approx, 0)
    return approx


def find_orthonormal_basis(dimension, X, given=None):
    if given is None:
        basis = np.identity(dimension)
    else:
        basis = given.copy()
    for i in range(dimension):
        products = np.zeros(dimension)
        for j in range(i):
            products[j] = get_inner_product(basis[:, j], basis[:, i], X)
        correction = np.dot(basis, products)
        orthogonal = basis[:, i] - correction
        basis[:, i] = normalize_on_distribution(orthogonal, X)
    return basis


def make_normal_indentity(dimension, X):
    basis = np.identity(dimension)
    for i in range(dimension):
        basis[:, i] = normalize_on_distribution(basis[:, i], X)
    return basis


def get_inner_product(v1, v2, X):
    first_values = np.dot(X, v1)
    second_values = np.dot(X, v2)
    average = np.average(np.multiply(first_values, second_values))
    return average


def normalize_on_distribution(vector, X):
    values = np.dot(X, vector)
    return vector / np.sqrt(np.average(np.square(values)))


def get_row_probabilities(matrix):
    size = len(matrix)
    probabilities = np.zeros(size)
    for i in range(size):
        norm = np.linalg.norm(matrix[i])
        probabilities[i] = norm * norm
    probabilities = probabilities / np.sum(probabilities)
    return probabilities


if __name__ == "__main__":
    import data_loader as dl
    from dtaidistance import dtw
    from msm import msm_fast
    from singular_values import calculate_best_relative_error_rank
    from aca import make_symmetrical

    """
    Start testing
    """
    data_name = "Crop"
    rank = 50
    epsilon = 2.0
    func_name = "dtw"
    debug = True
    start_index = None

    names = dl.get_all_dataset_names()
    size = len(names)
    index = 0
    try:
        errors = np.load("results/all_sets_solrad_" + str(func_name) + "_" + str(epsilon) + "_errors.npy")
        percs = np.load("results/all_sets_solrad_" + str(func_name) + "_" + str(epsilon) + "_percs.npy")
    except:
        errors = np.zeros(size)
        percs = np.zeros(size)

    sample_factors = np.zeros(size)
    for i in range(index, size):
        data_name = names[i]
        print("TESTING DATASET: " + data_name)
        # Read data from csv
        my_data = dl.read_train_and_test_data(data_name, debug=debug)
        print("Size: " + str(len(my_data)))

        solved_matrix = dl.load_array(data_name, func_name)

        # Create the problem class with internally the samplable matrix
        problem = cp.ClusterProblem(my_data, dtw.distance_fast, solved_matrix=solved_matrix)

        approx = solrad(problem, rank, epsilon=epsilon, debug=debug)

        percs[i] = problem.percentage_sampled()
        print("Percentage sampled = " + str(percs[i]))

        sample_factors[i] = percs[i] * problem.get_max_sample_amount() / problem.cp_size()
        print("Sample factor = " + str(sample_factors[i]))

        errors[i] = problem.get_relative_error(approx)
        print("Error = " + str(errors[i]))

        print(" & " + str(round(errors[i], 4)) + " & " + str(round(100 * percs[i], 2)) + " & " + str(
            round(sample_factors[i], 1)))

        np.save("results/all_sets_solrad_" + str(func_name) + "_" + str(epsilon) + "_percs", percs)
        np.save("results/all_sets_solrad_" + str(func_name) + "_" + str(epsilon) + "_errors", errors)
