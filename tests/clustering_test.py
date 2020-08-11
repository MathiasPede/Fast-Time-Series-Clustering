import data_loader as dl
import clustering as cl
from aca import aca_symm
from cluster_problem import ClusterProblem
from msm import msm_fast
from dtaidistance.dtw import distance_fast
from ed import ed_fast
from solrad import solrad
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ars
from plotter import scatter_plot, multiple_plot, multiple_scatter_plot

funcs = {"msm": msm_fast,
         "dtw": distance_fast,
         "ed": ed_fast
         }


def get_real_labels(cp, cluster_nbs):
    test_size = len(cluster_nbs)
    real_labels_agglo = np.zeros((test_size, cp.cp_size()))
    real_labels_spectral = np.zeros((test_size, cp.cp_size()))

    print("Calculating best labels")
    for i in range(test_size):
        real_labels_agglo[i] = cl.agglomerative(cp.solved_matrix, cluster_nbs[i], linkage="complete")
        real_labels_spectral[i] = cl.spectral(cp.solved_matrix, cluster_nbs[i])
    return real_labels_agglo, real_labels_spectral


def test_error_on_cluster_performance_aca(cp, tolerance, cluster_nbs, real_labels_agglo, real_labels_spectral, compare_func=ars):
    test_size = len(cluster_nbs)

    print("Calculating approximation")
    approx = aca_symm(cp, tolerance=tolerance)
    error = cp.get_relative_error(approx)
    print("ERROR: " + str(error))

    scores_agglo = np.zeros(len(cluster_nbs))
    scores_spectral = np.zeros(len(cluster_nbs))
    print("Comparing")
    for i in range(test_size):
        scores_agglo[i] = cl.compare_agglomerative(approx, cluster_nbs[i], real_labels_agglo[i], linkage="complete", compare_func=compare_func)
        scores_spectral[i]= cl.compare_spectral(approx, cluster_nbs[i], real_labels_spectral[i], compare_func=compare_func)
    return scores_agglo, scores_spectral, error


def test_error_on_cluster_performance_solrad(cp, epsilon, rank, cluster_nbs, real_labels_agglo, real_labels_spectral, compare_func=ars):
    test_size = len(cluster_nbs)

    print("Calculating approximation")
    approx = solrad(cp, rank, epsilon=epsilon)
    perc = cp.percentage_sampled()
    error = cp.get_relative_error(approx)
    print("ERROR: " + str(error))
    print("PERC: " + str(round(perc, 4)))

    scores_agglo = np.zeros(len(cluster_nbs))
    scores_spectral = np.zeros(len(cluster_nbs))
    for i in range(test_size):
        print("Comparing")
        scores_agglo[i] = cl.compare_agglomerative(approx, cluster_nbs[i], real_labels_agglo[i], linkage="complete", compare_func=compare_func)
        scores_spectral[i]= cl.compare_spectral(approx, cluster_nbs[i], real_labels_spectral[i], compare_func=compare_func)
    return scores_agglo, scores_spectral


def show_results(data_name, func_name):
    agglo_results = np.load("results/cluster_agglo_" + str(name) + "_" + func + ".npy")
    spectral_results = np.load("results/cluster_spectral_" + str(name) + "_" + func + ".npy")
    errors = np.load("results/cluster_errors_" + str(name) + "_" + func + ".npy")

    multiple_scatter_plot((agglo_results[:,0], agglo_results[:,1], agglo_results[:,2]),
                  (errors, errors, errors), ("k = 5", "k = 10", "k = 15"),
                  yname="Adjusted Rand Index", xname="Relatieve fout van benadering",
                  title="Spectraal performatie op basis van kwaliteit benadering", yscale='linear', regression=True)
    multiple_scatter_plot((spectral_results[:,0], spectral_results[:,1], spectral_results[:,2]),
                  (errors, errors, errors), ("k = 5", "k = 10", "k = 15"),
                  yname="Adjusted Rand Index", xname="Relatieve fout van benadering",
                  title="HAC performatie op basis van kwaliteit benadering", yscale='linear', regression=True)


def test_aca_all_datasets(tolerance, cluster_nbs, compare_func=ars, index=0):
    datasets = dl.get_all_dataset_names()
    size = len(datasets)
    test_size = len(cluster_nbs)

    try:
        results_nmi_agglo = np.load("results/all_datasets_cluster_agglo_" + func + ".npy")
        results_nmi_spectral = np.load("results/all_datasets_cluster_spectral_" + func + ".npy")
        result_errors = np.load("results/all_datasets_cluster_errors_" + func + ".npy")
        result_percs = np.load("results/all_datasets_cluster_percs_" + func + ".npy")
    except:
        result_percs = np.zeros(size)
        result_errors = np.zeros(size)
        results_nmi_agglo = np.zeros((size, test_size))
        results_nmi_spectral = np.zeros((size, test_size))
    for i in range(index, size):
        name = datasets[i]
        data = dl.read_train_and_test_data(name)
        print("Testing " + str(name) + " Size: " + str(len(data)))
        if len(data) < 800:
            print("Size too small")
            continue
        matrix = dl.load_array(name, func)
        cp = ClusterProblem(data, funcs[func], solved_matrix=matrix)

        real_labels_agglo, real_labels_spectral = get_real_labels(cp, cluster_nbs)

        results_nmi_agglo[i], results_nmi_spectral[i], result_errors[i] = test_error_on_cluster_performance_aca(cp,
                   tolerance, cluster_nbs, real_labels_agglo, real_labels_spectral, compare_func=compare_func)
        result_percs[i] = cp.percentage_sampled()
        print("PERC: " + str(round(result_percs[i], 4)))

        np.save("results/all_datasets_cluster_agglo_" + func, results_nmi_agglo)
        np.save("results/all_datasets_cluster_spectral_" + func, results_nmi_spectral)
        np.save("results/all_datasets_cluster_errors_" + func, result_errors)
        np.save("results/all_datasets_cluster_percs_" + func, result_percs)


if __name__ == '__main__':
    name = "ChlorineConcentration"
    func = "dtw"
    tolerance = 0.01
    cluster_nbs = (5, 10, 15)
    comp_func = nmi
    index = 13

    test_aca_all_datasets(tolerance, cluster_nbs, compare_func=comp_func)

    #show_results(name, func)
    # # Fully sampled matrix to compare with
    # data = dl.read_train_and_test_data(name)
    # matrix = dl.load_array(name, func)
    # cp = ClusterProblem(data, funcs[func], solved_matrix=matrix)
    #
    # real_labels_agglo, real_labels_spectral = get_real_labels(cp, cluster_nbs)
    #
    # agglo_results = []
    # spectral_results = []
    # errors = []
    # while tolerance > 0.005:
    #     print("TOLERANCE = " + str(tolerance))
    #     result_agglo, result_spectral, error = test_error_on_cluster_performance_aca(cp, tolerance, cluster_nbs,
    #                                                                                  real_labels_agglo, real_labels_spectral, compare_func=comp_func)
    #     agglo_results.append(result_agglo)
    #     spectral_results.append(result_spectral)
    #     errors.append(error)
    #     tolerance *= 0.85
    #
    # agglo_results = np.array(agglo_results)
    # spectral_results = np.array(spectral_results)
    # errors = np.array(errors)
    #
    # np.save("results/cluster_agglo_" + str(name) + "_" + func, agglo_results)
    # np.save("results/cluster_spectral_" + str(name) + "_" + func, spectral_results)
    # np.save("results/cluster_errors_" + str(name) + "_" + func, errors)
    #
    # show_results(data, func)


