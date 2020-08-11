import data_loader as dl
from cluster_problem import ClusterProblem
from msm import msm_fast
from dtaidistance.dtw import distance_fast
from ed import ed_fast
from aca import aca_symmetric_body, calc_symmetric_matrix_approx
import numpy as np
from plotter import multiple_plot

funcs = {"msm": msm_fast,
         "dtw": distance_fast,
         "ed": ed_fast
         }


if __name__ == '__main__':
    name = "ElectricDevices"
    func = "dtw"
    tolerance = 0.0001
    seed = 5

    # Fully sampled matrix to compare with
    data = dl.read_train_and_test_data(name)
    matrix = dl.load_array(name, func)
    cp = ClusterProblem(data, funcs[func], solved_matrix=matrix)

    testing_ranks = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,360,400,440,480,520], dtype=int)
    size = len(testing_ranks)
    real_errors = np.zeros(size)
    estimated_errors = np.zeros(size)

    for i in range(size):
        rows, deltas, m, error = aca_symmetric_body(cp, max_rank=testing_ranks[i], tolerance=tolerance, seed=seed)
        approx = calc_symmetric_matrix_approx(rows, deltas, cp.cp_size(), m)
        np.fill_diagonal(approx, 0)
        estimated_errors[i] = error
        real_errors[i] = cp.get_relative_error(approx)
        cp.reset_matrix()

    multiple_plot((real_errors, estimated_errors), (testing_ranks, testing_ranks), ("Echte fout", "Schatting fout"), xname='Rang', yname="Relatieve fout", title='Ingeschatte fout van ACA voor stopcriterium')