from __future__ import division, print_function
import traceback
from scipy.sparse import issparse
from sklearn import clone
import sys

from sklearn.model_selection import ParameterSampler
from sklearn.svm import SVC

from new_experiment_runner.cacher import CSVCacher
from new_experiment_runner.runner import Runner

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
from collections import OrderedDict, namedtuple
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

estimator_tuple = namedtuple('estimator_tuple',
                             ['name', 'estimator', 'kwargs_func', 'grid_func'])


def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


def split_datasets(X, y, X1, X2, z, Xu, n_splits, test_size=0.2):
    y_split = []
    z_split = []
    u_split = []
    if len(y) > 0:
        y_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size).split(X, y)
    if len(z) > 0:
        z_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size).split(X1, z)
    if len(Xu) > 0:
        u_split = ShuffleSplit(n_splits=n_splits, test_size=test_size).split(Xu, np.arange(len(Xu)))
    for (tr_y, te_y), (tr_z, te_z), (tr_u, te_u) in \
            itertools.izip_longest(y_split,
                                   z_split,
                                   u_split,
                                   fillvalue=(np.zeros(0, dtype=int), np.zeros(0, dtype=int))):
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


def links_grid_rbf(X, y, fit_kwargs):
    from links_vs_npklr_vs_svm import get_alpha_distribution, get_beta_distribution, \
        get_delta_distribution
    grid = {
        'alpha': get_alpha_distribution(2, len(y)),
        'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    }
    if 'z' in fit_kwargs and len(fit_kwargs['z']) > 0:
        grid['beta'] = get_beta_distribution(2, len(fit_kwargs['z']))
    if 'Xu' in fit_kwargs and len(fit_kwargs['Xu']) > 0:
        grid['delta'] = get_delta_distribution(2, len(fit_kwargs['Xu']))
    return grid


def svm_grid_rbf(X, y, fit_kwargs):
    from links_vs_npklr_vs_svm import get_alpha_distribution
    return {
        'C': get_alpha_distribution(2, len(y)),
        'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    }


def task(context, **kwargs):
    from start_sensitivity import split_dataset_stable
    from new_experiment_runner.cacher import CSVCacher
    from links_vs_npklr_vs_svm import split_datasets, accuracy_scorer
    from sklearn.model_selection import ParameterSampler
    from sklearn.base import clone
    estimator_tuple = kwargs.pop('estimator')

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
        disjoint_labels_and_links=False,
        random_state=42)

    # if len(cacher.get(context) > 1):
    #     continue



    rs_cacher = CSVCacher(filename=None)
    rs_context = {}

    for i_inner_split, ((tr_y, tr_z, tr_u), (te_y, te_z, te_u)) \
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
        rs_context['rs_split'] = i_inner_split

        fit_kwargs = {
            'X1': X1_tr[tr_z],
            'X2': X2_tr[tr_z],
            'z': z_tr[tr_z],
            'Xu': Xu_tr[tr_u],
        }
        fit_kwargs = estimator_tuple.kwargs_func(fit_kwargs)
        grid = estimator_tuple.grid_func(X_tr[tr_y],
                                         y_tr[tr_y], fit_kwargs)

        for params in ParameterSampler(grid, context['rs_iters'], random_state=42):
            rs_context.update(params)

            # print(gs_context)

            estimator0 = clone(estimator_tuple.estimator)
            estimator0.set_params(**params)
            estimator0.fit(X_tr[tr_y],
                           y_tr[tr_y],
                           **fit_kwargs)
            score = accuracy_scorer(estimator0, X_tr[te_y], y_tr[te_y])

            rs_cacher.set(rs_context, {'score': score})

    rs_df = rs_cacher.dataframe
    # print(rs_df.shape)
    param_names = grid.keys()
    grouped = rs_df['score']. \
        groupby(by=map(lambda param_name: rs_df[param_name], param_names)).mean()
    best_params = grouped.argmax()
    # print(best_params)
    cv_score = grouped.ix[best_params]

    best_params = {name: best_params[i] for i, name in enumerate(param_names)}
    estimator_best = clone(estimator_tuple.estimator)
    estimator_best.set_params(**best_params)
    # estimator_best.verbose = True
    # print(context, 'fitting on full train set')
    estimator_best.fit(X_tr, y_tr, X1=X1_tr, X2=X2_tr, z=z_tr, Xu=Xu_tr)

    test_score = accuracy_scorer(estimator_best, X[test], y[test])

    result = dict(best_params)
    result['cv_score'] = cv_score
    result['test_score'] = test_score
    return result


def validate_percents(X, y, p_labels, p_links, p_unlabeled, disjoint=False):
    try:
        _ = split_dataset_stable(X, y, p_labels, p_links, p_unlabeled,
                                 disjoint_labels_and_links=disjoint,
                                 return_index=True)
        return True
    except:
        return False


if __name__ == '__main__':

    mp.freeze_support()

    datafiles_toy = [
        r'data/diabetes_scale.libsvm',
        r'data/breast-cancer_scale.libsvm',
        r'data/australian_scale.libsvm',
        r'data/ionosphere_scale.libsvm',
        r'data/german.numer_scale.libsvm',
        r'data/heart_scale.libsvm',
        r'data/liver-disorders_scale.libsvm'

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

    context = OrderedDict(
        rs_test_size=args.rs_test_size,
        rs_splits=args.rs_folds,
        cv_test_size=args.cv_test_size,
        cv_splits=args.cv_folds,
        rs_iters=args.rs_iters,
        cv_random_state=42)

    percent_labels_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    percent_links_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    percent_unlabeled_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]


    def labels_only(fit_kwargs):
        fit_kwargs.pop('X1')
        fit_kwargs.pop('X2')
        fit_kwargs.pop('z')
        fit_kwargs.pop('Xu')
        return fit_kwargs


    def labels_links(fit_kwargs):
        fit_kwargs.pop('Xu')
        return fit_kwargs


    estimators = [
        estimator_tuple(
            name='Links',
            estimator=LinksClassifier(kernel='rbf', sampling='predefined', solver='tnc'),
            kwargs_func=lambda kw: kw,
            grid_func=links_grid_rbf),
        estimator_tuple(
            name='NPKLR',
            estimator=LogisticRegressionPairwise(kernel='rbf', sampling='predefined'),
            kwargs_func=labels_links,
            grid_func=links_grid_rbf),
        estimator_tuple(
            name='SVM',
            estimator=SVC(kernel='rbf'),
            kwargs_func=labels_only,
            grid_func=svm_grid_rbf)
    ]


    def task_generator():
        for ds_name, (X, y) in datasets.iteritems():
            if issparse(X):
                X = X.toarray()
            context['dataset'] = ds_name

            for estimator_tuple in estimators:
                context['estimator'] = estimator_tuple.name

                if isinstance(estimator_tuple.estimator, LinksClassifier):
                    links_and_labels = itertools.izip(percent_labels_range, percent_links_range)
                    percents = [(links, labels, unlabeled) for (links, labels), unlabeled in
                                itertools.product(links_and_labels, percent_unlabeled_range)]
                elif isinstance(estimator_tuple.estimator, LogisticRegressionPairwise):
                    percents = itertools.izip_longest(percent_labels_range,
                                                      percent_links_range,
                                                      [], fillvalue=0.0)
                elif isinstance(estimator_tuple.estimator, SVC):
                    percents = itertools.izip_longest(percent_labels_range,
                                                      [],
                                                      [], fillvalue=0.0)
                else:
                    raise Exception(
                        "I don't know what to do with %s" % type(estimator_tuple.estimator))
                percents = list(percents)

                for p_labels, p_links, p_unlabeled in percents:

                    if not validate_percents(X, y, p_labels, p_links, p_unlabeled, disjoint=False):
                        continue

                    context['percent_labels'] = p_labels
                    context['percent_links'] = p_links
                    context['percent_unlabeled'] = p_unlabeled

                    outer_cv = StratifiedShuffleSplit(n_splits=args.cv_folds,
                                                      test_size=args.cv_test_size,
                                                      random_state=context[
                                                          'cv_random_state']).split(X, y)
                    for i_split, (train, test) in enumerate(outer_cv):
                        context['cv_split'] = i_split
                        yield OrderedDict(context), {
                            'estimator': estimator_tuple,
                            'X': X,
                            'y': y,
                            'train': train,
                            'test': test
                        }


    if args.jobs == 1:
        mapper = itertools.imap
    else:
        mapper = mp.Pool(mp.cpu_count() if args.jobs == -1 else args.jobs).imap_unordered

    runner = Runner(task=task, task_generator=task_generator(), cacher=cacher, mapper=mapper)
    runner.run()
