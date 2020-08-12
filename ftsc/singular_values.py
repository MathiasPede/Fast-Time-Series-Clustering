import numpy as np
from math import sqrt


def calculate_best_relative_error_rank(sing_vals, rank):
    if rank > len(sing_vals):
        return 0.0
    size = len(sing_vals)
    squared = np.square(sing_vals)
    absolute_error = sqrt(np.sum(squared[::-1][0:size-rank]))
    norm = sqrt(np.sum(squared))
    return absolute_error / norm


def calculate_best_relative_error_for_all_ranks(sing_vals):
    squared = np.square(sing_vals)
    norm = sqrt(np.sum(squared))
    abs_error = np.sqrt(np.cumsum(squared[::-1])[::-1])
    return abs_error / norm


def get_first_rank_relative_error_below(sing_vals, rel_error):
    rel_errors = calculate_best_relative_error_for_all_ranks(sing_vals)
    index = np.where(rel_errors < rel_error)[0][0]
    return index + 1
