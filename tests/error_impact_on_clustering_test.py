if __name__ == "__main__":
    import numpy as np
    import ftsc.clustering as clustering
    from ftsc.aca import aca_symm

    from tests.tests_utils import create_cluster_problem, get_labels, get_amount_of_classes
    from tests.plotting_utils import scatter_plot

    """
    Start testing
    """
    data_name = "Symbols"
    func_name = "dtw"
    test_size = 5
    size = 100
    total_tests = test_size * size
    start_tolerance = 0.10

    index = 0

    errors = np.zeros(total_tests)
    approx_scores = np.zeros((total_tests, 3))

    # Read data from csv
    ground_truth = get_labels(data_name)
    class_nb = get_amount_of_classes(ground_truth)
    print("Classes size: " + str(class_nb))

    problem = create_cluster_problem(data_name, func_name)
    solved_matrix = problem.sample_full_matrix()

    best_scores = np.zeros(3)
    best_scores[0] = clustering.compare_agglomerative(solved_matrix, class_nb, ground_truth)
    best_scores[1] = clustering.compare_spectral(solved_matrix, class_nb, ground_truth)
    eps = 2.5
    min_samples = 10
    labels = clustering.dbscan(solved_matrix, eps, min_samples=min_samples)
    best_scores[2] = clustering.compare_dbscan(solved_matrix, eps, ground_truth, min_samples=min_samples)
    print("Best scores: AGGLO: " + str(best_scores[0]) + " SPECTRAL: " + str(best_scores[1]) + " DBSCAN: " + str(best_scores[2]))

    current_tolerance = start_tolerance
    for i in range(size):
        print("Testing tolerenca: " + str(current_tolerance))
        for j in range(test_size):

            approx = aca_symm(problem, tolerance=current_tolerance)
            errors[index] = problem.get_relative_error(approx)
            print("Error = " + str(errors[index]))

            index = i*test_size + j

            approx_scores[index, 0] = clustering.compare_agglomerative(approx, class_nb, ground_truth)
            approx_scores[index, 1] = clustering.compare_spectral(approx, class_nb, ground_truth)
            approx_scores[index, 2] = clustering.compare_dbscan(approx, eps, ground_truth, min_samples=min_samples)

            print("Approx scores: AGGLO: " + str(approx_scores[index, 0]) + " SPECTRAL: " + str(approx_scores[index, 1])
                  + " DBSCAN: " + str(approx_scores[index,2]))

        current_tolerance *= 0.95

    diff = best_scores - approx_scores

    scatter_plot(diff[:,0], errors, yscale='linear', marker="o",
                 xname="Relatieve fout benadering", yname="Echte ARI - Benadering ARI",
                 title="HAC: Impact van relatieve fout op het ARI-verschil", regression=True)
    scatter_plot(diff[:,1], errors, yscale='linear', marker="o",
                 xname="Relatieve fout benadering", yname="Echte ARI - Benadering ARI",
                 title="Spectraal: Impact van relatieve fout op het ARI-verschil", regression=True)
    scatter_plot(diff[:,2], errors, yscale='linear', marker="o",
                 xname="Relatieve fout benadering", yname="Echte ARI - Benadering ARI",
                 title="DBSCAN: Impact van relatieve fout op het ARI-verschil", regression=True)



