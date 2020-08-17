import numpy as np
from ftsc.aca import calc_symmetric_matrix_approx, aca_symmetric_body
from tests.tests_utils import create_cluster_problem
from tests.plotting_utils import multiple_plot

if __name__ == '__main__':
    name = "ElectricDevices"
    func = "msm"
    tolerance = 0.0001

    # Fully sampled matrix to compare with
    cp = create_cluster_problem(name, func)

    testing_ranks = np.array(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380,
         400],
        dtype=int)
    size = len(testing_ranks)
    real_errors = np.zeros(size)
    estimated_errors = np.zeros(size)

    for i in range(size):
        rows, deltas, m, error = aca_symmetric_body(cp, max_rank=testing_ranks[i], tolerance=tolerance)
        approx = calc_symmetric_matrix_approx(rows, deltas, m)
        np.fill_diagonal(approx, 0)
        estimated_errors[i] = error
        real_errors[i] = cp.get_relative_error(approx)
        cp.reset_matrix()

    multiple_plot((real_errors, estimated_errors), (testing_ranks, testing_ranks), ("Echte fout", "Schatting fout"),
                  xname='Rang', yname="Relatieve fout", title='Ingeschatte fout van ACA voor stopcriterium')
