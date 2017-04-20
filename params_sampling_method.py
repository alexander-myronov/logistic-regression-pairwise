from __future__ import division, print_function
import traceback
from scipy.sparse import issparse
from sklearn import clone
import sys

from sklearn.model_selection import ParameterSampler

from new_experiment_runner.cacher import CSVCacher

__author__ = 'myronov'

# coding: utf-8

# In[1]:
import argparse
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import itertools
from collections import OrderedDict
from functools import partial

from sklearn.datasets import load_svmlight_file, make_circles, make_moons

from links import LinksClassifier
from logit import LogisticRegressionPairwise, LogisticRegression

from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit, GridSearchCV, \
    fit_grid_point, ShuffleSplit

from tqdm import tqdm as tqdm

from start_sensitivity import split_dataset_stable
import multiprocess as mp

from scipy.stats import expon


def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


def split_datasets(X, y, X1, X2, z, Xu, n_splits, test_size=0.2):
    y_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size).split(X, y)
    z_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size).split(X1, z)
    u_split = ShuffleSplit(n_splits=n_splits, test_size=test_size).split(Xu, np.arange(len(Xu)))
    for (tr_y, te_y), (tr_z, te_z), (tr_u, te_u) in itertools.izip(y_split, z_split, u_split):
        yield (tr_y, tr_z, tr_u), (te_y, te_z, te_u)


def get_alpha_distribution(method, n):
    if method == 1:
        return expon(1 / n, n)  # 'exp(n)'
    if method == 2:
        return expon(1 / (10 * n), 10 * n)
    if method == 3:
        return expon(1 / n, n)


def get_beta_distribution(method, n):
    if method == 1:
        return expon(1 / n, n)
    if method == 2:
        return expon(1 / (5 * n), 5 * n)
    if method == 3:
        return expon(1 / n, n)


def get_delta_distribution(method, n):
    if method == 1:
        return expon(1 / n, n)
    if method == 2:
        return expon(1 / n, n)
    if method == 3:
        return expon(1, 1)


def task(context, **kwargs):
    try:
        X = kwargs.pop('X')
        y = kwargs.pop('y')
        train = kwargs.pop('train')
        test = kwargs.pop('test')
        X_train, y_train = X[train], y[train]

        X_tr, y_tr, X1_tr, X2_tr, z_tr, Xu_tr = split_dataset_stable(
            X_train, y_train,
            percent_labels=context['percent_labels'],
            percent_links=context['percent_links'],
            percent_unlabeled=context['percent_unlabeled'],
            labels_and_links_separation_degree=False,
            random_state=42)

        estimator = LinksClassifier(sampling='predefined',
                                    init='normal',
                                    verbose=False,
                                    solver='tnc',
                                    kernel='rbf')
        # if len(cacher.get(context) > 1):
        #     continue



        rs_cacher = CSVCacher(filename=None)
        rs_context = {}

        for i_inner_split, ((tr_y, tr_z, tr_u), (te_y, tr_z, tr_u)) \
                in enumerate(
            split_datasets(
                X_tr,
                y_tr,
                X1_tr,
                X2_tr,
                z_tr,
                Xu_tr,
                n_splits=context['rs_splits'],
                test_size=context['rs_test_size'])):
            rs_context['re_split'] = i_inner_split

            grid = {
                'alpha': get_alpha_distribution(context['sampling_method'], len(y_tr)),
                'beta': get_beta_distribution(context['sampling_method'], len(z_tr)),
                'delta': get_delta_distribution(context['sampling_method'], len(Xu_tr)),
                'gamma': [0.01, 0.05, 0.1, 0.5, 1, 2],
                # 'kernel': ['rbf'],
            }

            for params in ParameterSampler(grid, context['rs_iters'], random_state=42):
                rs_context.update(params)

                # print(gs_context)

                estimator0 = clone(estimator)
                estimator0.set_params(**params)
                estimator0.fit(X_tr[tr_y],
                               y_tr[tr_y],
                               X1=X1_tr[tr_z],
                               X2=X2_tr[tr_z],
                               z=z_tr[tr_z],
                               Xu=Xu_tr[tr_u])
                score = accuracy_scorer(estimator0, X_tr[te_y], y_tr[te_y])

                rs_cacher.set(rs_context, {'score': score})

        rs_df = rs_cacher.dataframe
        #print(rs_df.shape)
        grouped = rs_df['score'].groupby(by=[rs_df['alpha'],
                                             rs_df['beta'],
                                             rs_df['delta'],
                                             rs_df['gamma']]).mean()
        best_params = grouped.argmax()
        # print(best_params)
        cv_score = grouped.ix[best_params]

        best_params = {
            'alpha': best_params[0],
            'beta': best_params[1],
            'delta': best_params[2],
            'gamma': best_params[3]}
        estimator_best = clone(estimator)
        estimator_best.set_params(**best_params)
        # estimator_best.verbose = True
        # print(context, 'fitting on full train set')
        estimator_best.fit(X_tr, y_tr, X1=X1_tr, X2=X2_tr, z=z_tr, Xu=Xu_tr)

        test_score = accuracy_scorer(estimator_best, X[test], y[test])

        return context, {
            'alpha': best_params['alpha'],
            'alpha/n': best_params['alpha'] / len(y_tr),
            'beta': best_params['beta'],
            'beta/n': best_params['beta'] / len(z_tr),
            'delta': best_params['delta'],
            'delta/n': best_params['delta'] / len(Xu_tr),
            'gamma': best_params['gamma'],
            'cv_score': cv_score,
            'test_score': test_score
        }
    except Exception as e:
        e_t, e_v, e_tb = sys.exc_info()
        e_tb = traceback.format_tb(e_tb)
        return context, (e_t, e_v, e_tb)


if __name__ == '__main__':

    mp.freeze_support()

    datafiles_toy = [
        r'data/diabetes_scale.libsvm',
        r'data/breast-cancer_scale.libsvm',
        r'data/australian_scale.libsvm',
        # r'data/ionosphere_scale.libsvm',
    ]


    def load_ds(fname):
        X, y = load_svmlight_file(fname)
        if issparse(X):
            X = X.toarray()
        y[y == -1] = 0
        return X, y


    datasets = OrderedDict([(os.path.split(f)[-1].replace('.libsvm', ''),
                             load_ds(f))
                            for f in datafiles_toy])
    datasets['circles'] = make_circles(n_samples=400, noise=0.1, factor=0.51)
    datasets['moons'] = make_moons(n_samples=400, noise=0.1)

    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='cross validation number of folds')
    parser.add_argument('--cv_test_size', type=float, default=0.2,
                        help='cross validation test size')

    parser.add_argument('--rs_folds', type=int, default=3,
                        help='random search number of folds')
    parser.add_argument('--rs_iters', type=int, default=25,
                        help='random search number of params combinations')
    parser.add_argument('--rs_test_size', type=float, default=0.2,
                        help='random search test size')

    parser.add_argument('--jobs', type=int, default=1,
                        help='number of parallel jobs, -1 for all')

    parser.add_argument('--file', type=str, default='data/results_semi.csv',
                        help='folder to store results')

    args = parser.parse_args()
    cacher = CSVCacher(filename=args.file)

    context = {'rs_test_size': args.rs_test_size,
               'rs_splits': args.rs_folds,
               'cv_test_size': args.cv_test_size,
               'cv_splits': args.cv_folds,
               'rs_iters': args.rs_iters,
               'cv_random_state': 42}

    percent_labels_range = [0.4]  # np.linspace(0.1, 0.3, 5)
    percent_links_range = [0.4]  # np.linspace(0.1, 0.3, 5)
    percent_unlabeled_range = [0.4]


    def task_generator():
        for ds_name, (X, y) in datasets.iteritems():
            if issparse(X):
                X = X.toarray()
            context['dataset'] = ds_name

            for sampling_method in [1, 2, 3]:
                context['sampling_method'] = sampling_method

                for (i_label, p_labels), (i_link, p_links), (i_unlableled, p_unlabeled) in \
                        itertools.product(enumerate(percent_labels_range),
                                          enumerate(percent_links_range),
                                          enumerate(percent_unlabeled_range)):
                    context['percent_labels'] = p_labels
                    context['percent_links'] = p_links
                    context['percent_unlabeled'] = p_unlabeled

                    outer_cv = StratifiedShuffleSplit(n_splits=args.cv_folds,
                                                      test_size=args.cv_test_size,
                                                      random_state=context[
                                                          'cv_random_state']).split(X, y)
                    for i_split, (train, test) in enumerate(outer_cv):
                        context['cv_split'] = i_split
                        if len(cacher.get(context)) == 0:
                            yield dict(context), {
                                'X': X,
                                'y': y,
                                'train': train,
                                'test': test
                            }


    if args.jobs == 1:
        mapper = itertools.imap
    else:
        mapper = mp.Pool(mp.cpu_count() if args.jobs == -1 else args.jobs).imap_unordered

    tasks = list(task_generator())
    tq = tqdm(total=len(tasks))


    def map_f(context_kwds):
        context, kwds = context_kwds
        from params_sampling_method import task
        return task(context, **kwds)


    for context, result in mapper(map_f, tasks):

        if len(result) == 3 and isinstance(result[1], Exception):
            print(result[0])
            print(result[1])
            print('\n'.join(result[2]))
            continue
        cacher.set(context, result)
        cacher.save()
        tq.update()
