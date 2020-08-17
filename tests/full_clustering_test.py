if __name__ == "__main__":
    import numpy as np

    import ftsc.clustering as clustering
    from ftsc.solradm import solradm

    from tests.plotting_utils import scatter_plot
    from tests.tests_utils import get_all_test_dataset_names, get_labels, create_cluster_problem, get_amount_of_classes
    """
    Start testing
    """
    rank = 5
    epsilon = 2.0
    func_name = "dtw"
    debug = False
    start_index = None

    names = get_all_test_dataset_names()
    size = len(names)
    index = 0

    errors = np.zeros(size)
    percs = np.zeros(size)
    best_scores = np.zeros((size, 2))
    approx_scores = np.zeros((size, 2))


    sample_factors = np.zeros(size)
    for i in range(index, size):
        data_name = names[i]
        print("TESTING DATASET: " + data_name)
        # Read data from csv
        ground_truth = get_labels(data_name)
        class_nb = get_amount_of_classes(ground_truth)
        print("Classes size: " + str(class_nb))

        problem = create_cluster_problem(data_name, func_name)
        solved_matrix = problem.sample_full_matrix()

        if best_scores[i, 0] == 0.0:
            best_scores[i, 0] = clustering.compare_agglomerative(solved_matrix, class_nb, ground_truth)
            best_scores[i, 1] = clustering.compare_spectral(solved_matrix, class_nb, ground_truth)

        print("Best scores: AGGLO: " + str(best_scores[i, 0]) + " SPECTRAL: " + str(best_scores[i, 1]))

        # Create the problem class with internally the samplable matrix
        approx = solradm(problem, rank, epsilon=epsilon)

        approx_scores[i, 0] = clustering.compare_agglomerative(approx, class_nb, ground_truth)
        approx_scores[i, 1] = clustering.compare_spectral(approx, class_nb, ground_truth)
        print("Approx scores: AGGLO: " + str(approx_scores[i, 0]) + " SPECTRAL: " + str(approx_scores[i, 1]))

        percs[i] = problem.percentage_sampled()
        print("Percentage sampled = " + str(percs[i]))

        sample_factors[i] = percs[i] * problem.get_max_sample_amount() / problem.cp_size()
        print("Sample factor = " + str(sample_factors[i]))

        errors[i] = problem.get_relative_error(approx)
        print("Error = " + str(errors[i]))

        print(" & " + str(round(errors[i], 4)) + " & " + str(round(100 * percs[i], 2)) + " & " + str(
            round(sample_factors[i], 1)))

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



