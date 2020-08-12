from ftsc.data_loader import load_timeseries_from_tsv
from ftsc.cluster_problem import ClusterProblem
from ftsc.distance_functions import compute_distance_matrix, compute_row
from ftsc.aca import aca_symm
import logging

logger = logging.getLogger("ftsc")
logger.setLevel(logging.WARNING)

name = "Symbols"
path = "tests/Data/" + name + "/" + name + "_TRAIN.tsv"
data = load_timeseries_from_tsv(path)
cp = ClusterProblem(data, "dtw")
#matrix = compute_row(data[:,1:], 23, "dtw")
approx = aca_symm(cp, tolerance=0.05, max_rank=10)