from ftsc.data_loader import load_timeseries_from_tsv
from ftsc.cluster_problem import ClusterProblem
from ftsc.distance_functions import compute_distance_matrix

name = "Symbols"
path = "tests/Data/" + name + "/" + name + "_TRAIN.tsv"
data = load_timeseries_from_tsv(path)
cp = ClusterProblem(data, "msm")
matrix = compute_distance_matrix(data[:,1:], "msm")