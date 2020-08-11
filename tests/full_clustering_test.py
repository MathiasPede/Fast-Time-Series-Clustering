if __name__ == "__main__":
    import data_loader as dl
    from dtaidistance import dtw
    import numpy as np
    import cluster_problem as cp
    from solrad import solrad
    import clustering
    from msm import msm_fast
    from singular_values import calculate_best_relative_error_rank
    from aca import make_symmetrical
    from plotter import scatter_plot
    """
    Start testing
    """
    rank = 5
    epsilon = 2.0
    func_name = "dtw"
    debug = False
    start_index = None

    names = dl.get_all_dataset_names()
    size = len(names)
    index = 0
    try:
        errors = np.load("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_errors.npy")
        percs = np.load("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_percs.npy")
        best_scores = np.load("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_best_scores.npy")
        approx_scores = np.load("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_approx_scores.npy")
    except:
        errors = np.zeros(size)
        percs = np.zeros(size)
        best_scores = np.zeros((size, 2))
        approx_scores = np.zeros((size, 2))


    sample_factors = np.zeros(size)
    # for i in range(index, size):
    #     data_name = names[i]
    #     print("TESTING DATASET: " + data_name)
    #     # Read data from csv
    #     my_data = dl.read_train_and_test_data(data_name, debug=debug)
    #     print("Size: " + str(len(my_data)))
    #     ground_truth = dl.get_cluster_labels(my_data)
    #     class_nb = dl.get_amount_of_classes(my_data)
    #     print("Classes size: " + str(class_nb))
    #
    #     solved_matrix = dl.load_array(data_name, func_name)
    #
    #     if best_scores[i, 0] == 0.0:
    #         best_scores[i, 0] = clustering.compare_agglomerative(solved_matrix, class_nb, ground_truth)
    #         best_scores[i, 1] = clustering.compare_spectral(solved_matrix, class_nb, ground_truth)
    #
    #     print("Best scores: AGGLO: " + str(best_scores[i, 0]) + " SPECTRAL: " + str(best_scores[i, 1]))
    #
    #     # Create the problem class with internally the samplable matrix
    #     problem = cp.ClusterProblem(my_data, dtw.distance_fast, solved_matrix=solved_matrix)
    #     approx = solrad(problem, rank, epsilon=epsilon, debug=debug)
    #
    #     approx_scores[i, 0] = clustering.compare_agglomerative(approx, class_nb, ground_truth)
    #     approx_scores[i, 1] = clustering.compare_spectral(approx, class_nb, ground_truth)
    #     print("Approx scores: AGGLO: " + str(approx_scores[i, 0]) + " SPECTRAL: " + str(approx_scores[i, 1]))
    #
    #     percs[i] = problem.percentage_sampled()
    #     print("Percentage sampled = " + str(percs[i]))
    #
    #     sample_factors[i] = percs[i] * problem.get_max_sample_amount() / problem.cp_size()
    #     print("Sample factor = " + str(sample_factors[i]))
    #
    #     errors[i] = problem.get_relative_error(approx)
    #     print("Error = " + str(errors[i]))
    #
    #     print(" & " + str(round(errors[i], 4)) + " & " + str(round(100 * percs[i], 2)) + " & " + str(
    #         round(sample_factors[i], 1)))
    #
    #     np.save("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_errors", errors)
    #     np.save("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_percs", percs)
    #     np.save("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_best_scores", best_scores)
    #     np.save("results/all_clustering_solrad_" + str(func_name) + "_" + str(rank) + "_approx_scores", approx_scores)

    average_error = np.average(errors)
    print("ERROR: " + str(average_error))

    diff = best_scores - approx_scores
    diff_divided = np.divide(approx_scores, best_scores)
    average = np.average(diff, axis=0)
    std = np.std(diff, axis=0)

    average_divided = np.average(diff_divided, axis=0)
    average_std = np.std(diff_divided, axis=0)
    print("AVERAGE: " + str(average))
    print("STD: " + str(std))

    print("AVERAGE DIV: " + str(average_divided))
    print("STD DIV: " + str(average_std))


    scatter_plot(approx_scores[:,0], best_scores[:, 0], yscale='linear', marker="o", xrange=(0.0,1.0), yrange=(0.0,1.0),
                 xname="ARI exacte afstandsmatrix", yname="ARI benaderde afstandsmatrix",
                 title="Vergelijking ARI scores voor complete linkage HAC (Rang = " + str(rank) + ")", regression=True)
    scatter_plot(approx_scores[:,1], best_scores[:, 1], yscale='linear', marker="o", xrange=(0.0,1.0), yrange=(0.0,1.0),
                 xname="ARI exacte afstandsmatrix", yname="ARI benaderde afstandsmatrix",
                 title="Vergelijking ARI scores voor spectraal clusteren (Rang = " + str(rank) + ")", regression=True)



