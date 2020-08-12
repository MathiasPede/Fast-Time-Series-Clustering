import numpy as np
import logging
from .cluster_problem import ClusterProblem

logger = logging.getLogger("ftsc")


def load_timeseries_from_tsv(path):
    """
    Loads Time Series from TSV file. The Format is expected to be the Class number as first element of the row,
    followed by the the elements of the time series.
    @param path:
    @return:
    """
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



