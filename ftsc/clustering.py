from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering, OPTICS
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ars
import numpy as np


def convert_to_similarity_matrix(distance_matrix, delta=0.5):
    simil_matrix = np.exp(- distance_matrix ** 2 / (2. * delta ** 2))
    return simil_matrix


def agglomerative(matrix, k, linkage='complete'):
    matrix = prepare_distance_matrix(matrix)
    model_agglo = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage=linkage)
    result_agglo = model_agglo.fit_predict(matrix)
    return result_agglo


def compare_agglomerative(matrix, k, real_classes, linkage='complete', compare_func=ars):
    results = agglomerative(matrix, k, linkage=linkage)
    score = compare_func(real_classes, results)
    return score


def optics(matrix):
    matrix = prepare_distance_matrix(matrix)
    model_optics = OPTICS()
    result_optics = model_optics.fit_predict(matrix)
    return result_optics


def compare_optics(matrix, real_classes, compare_func=ars):
    results = optics(matrix)
    score = compare_func(real_classes, results)
    return score


def dbscan(matrix, eps, min_samples=5):
    matrix = prepare_distance_matrix(matrix)
    model_optics = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    result_dbscan = model_optics.fit_predict(matrix)
    return result_dbscan


def compare_dbscan(matrix, eps, real_classes, min_samples=5, compare_func=ars):
    results = dbscan(matrix, eps, min_samples=min_samples)
    score = compare_func(real_classes, results)
    return score


def spectral(matrix, k):
    matrix = prepare_distance_matrix(matrix)
    delta = matrix.max() - matrix.min()
    simil_matrix = convert_to_similarity_matrix(matrix, delta=delta)
    model_agglo = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0)
    result_agglo = model_agglo.fit_predict(simil_matrix)
    return result_agglo


def compare_spectral(matrix, k, real_classes, compare_func=ars):
    results = spectral(matrix, k)
    score = compare_func(real_classes, results)
    return score


# Fix small negative distances
def prepare_distance_matrix(matrix):
    return abs(matrix)