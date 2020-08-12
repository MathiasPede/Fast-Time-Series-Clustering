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
    labels, series = data[:, 0], data[:, 1:]
    return labels, series


def load_timeseries_from_multiple_tsvs(*args):
    all_labels = []
    all_series = []
    for path in args:
        labels, series = load_timeseries_from_tsv(path)
        all_labels.append(labels)
        all_series.append(series)
    result_labels = np.concatenate(all_labels)
    result_series = np.concatenate(all_series)
    return result_labels, result_series



