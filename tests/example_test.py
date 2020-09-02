from ftsc.data_loader import load_timeseries_from_multiple_tsvs
from ftsc.cluster_problem import ClusterProblem
from ftsc.solradm import solradm
import time

name = "ECG5000"
path1 = "Data/" + name + "/" + name + "_TRAIN.tsv"
path2 = "Data/" + name + "/" + name + "_TEST.tsv"

labels, series = load_timeseries_from_multiple_tsvs(path1, path2)
cp = ClusterProblem(series, "dtw")

start_time = time.time()
approx = solradm(cp, 50, epsilon=2.0)
end_time = time.time()
print("Time spent on approximation: " + str(end_time - start_time) + " seconds")

start_time = time.time()
cp.sample_full_matrix()
end_time = time.time()
print("Time spent on exact matrix: " + str(end_time - start_time) + " seconds")

relative_error = cp.get_relative_error(approx)
print("Relative error of the approximation: " + str(relative_error))


