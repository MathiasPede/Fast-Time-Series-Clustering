"""
Similarity Preserving Representation Learning (SPRL) for Time Series Clustering
"""

import ftsc.cluster_problem as cp
import numpy as np
import random as rnd
from math import log, floor, pow, sqrt, atan2, cos


def spiral(clustp, k, sample_factor=20, max_iterations=20):
    if sample_factor is None:
        sample_factor = 20
    n = clustp.cp_size()
    s = floor(sample_factor * n * log(n))

    clustp.sample_diagonal()

    for i in range(s):
        x = rnd.randint(0, n-1)
        y = rnd.randint(0, n-1)

        clustp.sample(x, y)

    clustp.compute_omega()

    X = eeccda(clustp, k, max_iterations=max_iterations)

    return np.matmul(X, np.transpose(X))


def eeccda(clustp, k, max_iterations=20):
    """
    Efficient Exact Cyclic Coordinate Descent Algorithm
    """
    n = clustp.cp_size()
    X = np.zeros(shape=(n, k))
    R = np.zeros(shape=(n, n))
    R = clustp.matrix

    old_residu_norm = get_projected_norm(R, clustp.compute_omega())
    for t in range(max_iterations):
        for i in range(k):
            R = np.add(R, np.outer(X[:, i], X[:, i]))
            for j in range(n):
                omega_indices = clustp.get_omega_row(j)
                X_omega = X[omega_indices, i]
                p = np.sum(np.square(X_omega))
                p -= X[j, i] * X[j, i] + R[j, j]
                q = - np.sum(np.multiply(X_omega, R[j, omega_indices])) - X[j, i] * R[j, j]

                X[j, i] = root(p, q)

            R = np.subtract(R, np.outer(X[:, i], X[:, i]))

        new_residu = get_projected_norm(R, clustp.compute_omega())
        if old_residu_norm / new_residu > 1.00001:
            old_residu_norm = new_residu
        else:
            print("Stop criterion achieved after " + str(t) + " iterations")
            break
    return X


def get_projected_norm(matrix, omega):
    row_norms = np.zeros(len(omega))
    for i in range(len(omega)):
        row_norms[i] = np.linalg.norm(matrix[i, omega[i]])

    squared = np.square(row_norms)
    summed = np.sum(squared)
    return sqrt(summed)


def cubic_root(d):
    if d < 0.0:
        return -cubic_root(-d)
    else:
        return pow(d, 1.0/3.0)


def root(a, b):
    x = 0
    y = 0
    a3 = 4 * pow(a, 3)
    b2 = 27 * pow(b, 2)
    delta = a3 + b2

    if delta <= 0.0:
        # 3 distint real roots or 1 real multiple solution
        r3 = 2*sqrt(-a/3)
        th3 = atan2(sqrt(-delta/108), -b/2)/3
        ymax = 0.0
        xopt = 0
        for k in range(0, 5, 2):
            x = r3*cos(th3+((k*3.14159265)/3))
            y = pow(x, 4)/4 + a*pow(x, 2)/2 + b*x
            if y < ymax:
                ymax = y
                xopt = x
        return xopt

    else:
        # /* 1 real root and two complex */
        z = sqrt(delta/27)
        x = cubic_root(0.5*(-b+z))+cubic_root(0.5*(-b-z))
        y = pow(x,4)/4+a*pow(x,2)/2+b*x
        return x


def convert_distance_matrix_to_similarity_matrix(distance_matrix, distances_with_zero):
    size = len(distance_matrix)
    similarity_matrix = np.zeros((size,size), dtype=float)
    for i in range(size):
        term1 = distances_with_zero[i]
        for j in range(i, size):
            term2 = distances_with_zero[j]
            term3 = distance_matrix[i,j]
            similarity_matrix[i,j] = (term1*term1 + term2*term2 - term3*term3) / (2.0 * term1 * term2)
            similarity_matrix[j,i] = similarity_matrix[i,j]
    return similarity_matrix


def convert_similarity_matrix_to_distance_matrix(similarity_matrix, distances_with_zero):
    size = len(similarity_matrix)
    distance_matrix = np.zeros((size,size), dtype=float)
    for i in range(size):
        term1 = distances_with_zero[i]
        for j in range(i, size):
            term2 = distances_with_zero[j]
            term3 = similarity_matrix[i,j]
            distance_matrix[i,j] = sqrt(abs((2.0 * term1 * term2 * term3) - term1*term1 - term2*term2))
            distance_matrix[j,i] = distance_matrix[i,j]
    return distance_matrix


def compute_distances_with_zero(data, func):
    size = len(data)
    distances = np.zeros(size)
    for i in range(size):
        distances[i] = func(data[i][1:], np.array([0], dtype=float))
    return distances

