from functools import partial
import sys

__author__ = 'amyronov'

import pandas as pd
import numpy as np

def transform(str_array, array_mapper):
    if isinstance(str_array, float):
        if np.isnan(str_array):
            return array_mapper([])
        return [str_array]
    if isinstance(str_array, str) and str_array[0] != '[':
        return str_array
    try:
        arr= np.fromstring(str_array.replace('[', '').replace(']',''), sep=' ')
        return array_mapper(arr)

    except:
        return str_array

def mean_std_tuple(arr):
    return arr.mean(), arr.std()

def str_mean_plus_minus_std(arr):
    return '%.3lf +- %.3lf' % (arr.mean(), arr.std())

if __name__ == '__main__':

    results_file = sys.argv[1]

    files = [
        'data/results_all/results.csv',
        'data/results_all/results_amazon.csv',
        'data/results_all/results_log.csv',
        'data/results_all/results_trzmiel.csv',
    ]

    df_res = None

    def concat(df, df2):
        for ds in df.index:
            #dsi = np.where(df['dataset'].values == ds)[0][0]
            for c in df.columns[1:]:
                v1 = df.get_value(ds, c)
                v2 = df2.get_value(ds, c)

                df.set_value(ds,c,np.concatenate([v1, v2]))

    for results_file in files:

        df = pd.read_csv(results_file)
        df.set_index(df['dataset'], inplace=True, drop=True)


        def amap(arr):
            if len(arr) > 3:
                return np.zeros(shape=0)
            return arr
        df_tr = df.applymap(partial(transform, array_mapper=amap))

        if df_res is None:
            df_res = df_tr
        else:
            concat(df_res, df_tr)

    df_res = df.applymap(partial(transform, array_mapper=str_mean_plus_minus_std))
    df_res.to_csv(r'data/result_processed.csv')
