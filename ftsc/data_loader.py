import numpy as np
import logging
from .cluster_problem import ClusterProblem

logger = logging.getLogger("ftsc")


def load_timeseries_from_tsv(path):
    logger.debug("Loading data from: " + path)
    data = np.genfromtxt(path, delimiter='\t')
    logger.debug("Loaded " + str(len(data)) + " data entries")
    return data


def load_timeseries_from_multiple_tsvs(*args):
    size = len(*args)
    all_data = []
    for i in range(size):
        data = load_timeseries_from_tsv(*args[i])
        all_data.append(data)
    result = np.concatenate(all_data)
    return result



