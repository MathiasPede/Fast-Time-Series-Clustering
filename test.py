from ftsc.data_loader import load_timeseries_from_tsv, load_timeseries_from_multiple_tsvs
from ftsc.cluster_problem import ClusterProblem
from ftsc.distance_functions import compute_distance_matrix, compute_row
from ftsc.aca import aca_symm
from ftsc.solradm import solradm
import logging
from tests.tests_utils import create_cluster_problem

name = "ECG5000"
path1 = "tests/Data/" + name + "/" + name + "_TRAIN.tsv"
path2 = "tests/Data/" + name + "/" + name + "_TEST.tsv"

labels, series = load_timeseries_from_multiple_tsvs(path1, path2)
cp = ClusterProblem(series, "dtw")

approx = aca_symm(cp, tolerance=0.05, max_rank=5)

percentage_sampled = cp.percentage_sampled()

# relative_error = cp.get_relative_error(approx)



