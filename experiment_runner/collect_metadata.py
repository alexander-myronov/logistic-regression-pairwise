from collections import defaultdict

import itertools
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from caching import MultipleFilesCacher

import numpy as np


def rmse_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def rmse_scorer(estimator, X, y):
    pass


def collect_metadata_dataframe(dir, naming_scheme=r'([0-9]+)_meta\.pkl'):
    cacher = MultipleFilesCacher(dir, file_match_regex=naming_scheme)
    items = cacher.items_dict
    df = pd.DataFrame(data=items.values(), index=items.keys())
    return df
