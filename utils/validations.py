import numpy as np


def test_if_necessary_columns_exists(data_columns, options):
    for el in options:
        if el not in data_columns:
            raise Exception(f'The Table should contain an "{el}" column')


def log_transform(x):
    """Return a dataframe with logarithm appliec"""
    x = x.apply(np.log10)
    return x.to_frame()