from __future__ import division, print_function

import imp
import re

from scipy.sparse import issparse
from sklearn.svm import SVC

from new_experiment_runner.cacher import CSVCacher
from new_experiment_runner.runner import Runner

__author__ = 'myronov'

# coding: utf-8

# In[1]:
import argparse

import numpy as np

import os
import itertools
from collections import OrderedDict, namedtuple

from sklearn.datasets import load_svmlight_file

from links import LinksClassifier
from logit import LogisticRegressionPairwise

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from start_sensitivity import split_dataset_stable
import multiprocess as mp

estimator_tuple = namedtuple('estimator_tuple',
                             ['name', 'estimator', 'kwargs_func', 'grid_func'])


def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


def adj_rand_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return adjusted_rand_score(y_true, y_pred)


def split_datasets(X, y, X1, X2, z, Xu, n_splits, test_size=0.2, labels_special_case=False):
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
        if labels_special_case:
            res = np.zeros(len(np.unique(y)), dtype=int)
            for i, y_val in enumerate(np.unique(y)):
                choice = np.random.choice(tr_y[y[tr_y] == y_val], 1)
                res[i] = choice
            tr_y = res
        yield (tr_y, tr_z, tr_u), (te_y, te_z, te_u)


def task(context, **kwargs):
    from start_sensitivity import split_dataset
    from new_experiment_runner.cacher import CSVCacher
    from links_vs_npklr_vs_svm import split_datasets, accuracy_scorer, adj_rand_scorer
    from sklearn.model_selection import ParameterSampler
    from sklearn.base import clone
    import numpy as np
    estimator_tuple = kwargs.pop('estimator')

    X = kwargs.pop('X')
    y = kwargs.pop('y')
    train = kwargs.pop('train')
    test = kwargs.pop('test')
    X_train, y_train = X[train], y[train]
    n_classes = len(np.unique(y))

    X_tr, y_tr, X1_tr, X2_tr, z_tr, Xu_tr = split_dataset(
        X_train, y_train,
        percent_labels=context['percent_labels'],
        percent_links=context['percent_links'],
        percent_unlabeled=context['percent_unlabeled'],
        labels_and_links_separation_degree=1)

    # if len(cacher.get(context) > 1):
    #     continue

    scorer = context['scorer']
    if scorer == 'accuracy':
        scorer = accuracy_scorer
    elif scorer == 'adj_rand':
        scorer = adj_rand_scorer
    else:
        raise Exception('unknown scorer')

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
            test_size=context['rs_test_size'],
            labels_special_case=context['percent_labels'] == -1
        )):
        rs_context['rs_split'] = i_inner_split

        fit_kwargs = {
            'X1': X1_tr[tr_z],
            'X2': X2_tr[tr_z],
            'z': z_tr[tr_z],
            'Xu': Xu_tr[tr_u],
            'n_classes': n_classes
        }
        fit_kwargs = estimator_tuple.kwargs_func(fit_kwargs)
        grid = estimator_tuple.grid_func(X_tr[tr_y],
                                         y_tr[tr_y], fit_kwargs)

        for params in ParameterSampler(grid, context['rs_iters']):
            rs_context.update(params)

            # print(gs_context)

            estimator0 = clone(estimator_tuple.estimator)
            estimator0.set_params(**params)
            estimator0.fit(X_tr[tr_y],
                           y_tr[tr_y],
                           **fit_kwargs)
            score = scorer(estimator0, X_tr[te_y], y_tr[te_y])

            rs_cacher.set(rs_context, {'score': score})

    rs_df = rs_cacher.dataframe
    # print(rs_df.shape)

    fit_kwargs = {
        'X1': X1_tr,
        'X2': X2_tr,
        'z': z_tr,
        'Xu': Xu_tr,
        'n_classes': n_classes
    }
    fit_kwargs = estimator_tuple.kwargs_func(fit_kwargs)
    grid = estimator_tuple.grid_func(X_tr,
                                     y_tr, fit_kwargs)

    param_names = grid.keys()
    grouped = rs_df['score']. \
        groupby(by=map(lambda param_name: rs_df[param_name], param_names)).mean()
    best_params = grouped.argmax()
    # print(grouped)
    cv_score = grouped.ix[best_params]
    grouped = None

    if not hasattr(best_params, '__iter__'):
        best_params = [best_params]
    best_params = {name: best_params[i] for i, name in enumerate(param_names)}
    estimator_best = clone(estimator_tuple.estimator)
    estimator_best.set_params(**best_params)
    # estimator_best.verbose = True
    # print(context, 'fitting on full train set')
    fit_kwargs = dict(X1=X1_tr, X2=X2_tr, z=z_tr, Xu=Xu_tr, n_classes=n_classes)
    fit_kwargs = estimator_tuple.kwargs_func(fit_kwargs)
    if context['percent_labels'] == -1:
        res = np.zeros(len(np.unique(y)), dtype=int)
        full_index = np.arange(len(y_tr))
        for i, y_val in enumerate(np.unique(y)):
            choice = np.random.choice(full_index[y_tr == y_val], 1)
            res[i] = choice
        y_tr = y_tr[res]
        X_tr = X_tr[res]
        # print(y_tr)
    estimator_best.fit(X_tr, y_tr, **fit_kwargs)

    test_score = scorer(estimator_best, X[test], y[test])

    result = dict(best_params)
    result['cv_score'] = cv_score
    result['test_score'] = test_score
    # print(estimator_tuple.name, cv_score, test_score)
    return result


def validate_percents(X, y, p_labels, p_links, p_unlabeled, disjoint=0):
    try:
        _ = split_dataset_stable(X, y, p_labels, p_links, p_unlabeled,
                                 labels_and_links_separation_degree=disjoint,
                                 return_index=True)
        return True
    except:
        return False


if __name__ == '__main__':

    mp.freeze_support()


    # datafiles_toy = [
    #     r'data/diabetes_scale.libsvm',
    #     r'data/breast-cancer_scale.libsvm',
    #     r'data/australian_scale.libsvm',
    #     r'data/ionosphere_scale.libsvm',
    #     r'data/german.numer_scale.libsvm',
    #     r'data/heart_scale.libsvm',
    #     r'data/liver-disorders_scale.libsvm'
    #
    # ]


    def load_ds(fname):
        n_features = None
        with open(fname, 'r') as svmlight_file:
            next(svmlight_file)
            next(svmlight_file)
            next(svmlight_file)
            n_features_line = next(svmlight_file)

            match = re.match('#\s([0-9]+)', n_features_line)
            if match:
                n_features = int(match.groups()[0])
        X, y = load_svmlight_file(fname, n_features=n_features)
        if issparse(X):
            X = X.toarray()
        y[y == -1] = 0
        return X, y


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

    parser.add_argument('--file', type=str, default='data/links_npklr_svm.csv',
                        help='folder to store results')

    parser.add_argument('--estimators_file', type=str,
                        help='python file with list of estimator_tuples called `estimators`')
    parser.add_argument('--datasets_file', type=str,
                        help='text file with .libsvm files to be used in the experiment')

    args = parser.parse_args()

    datasets_file = args.datasets_file
    with open(datasets_file, 'r') as f:
        datafiles = [s.strip() for s in f.readlines()]

    datasets = OrderedDict([(os.path.split(f)[-1].replace('.libsvm', ''),
                             load_ds(f))
                            for f in datafiles])

    cacher = CSVCacher(filename=args.file)

    # no labels config

    context = OrderedDict(
        rs_test_size=args.rs_test_size,
        rs_splits=args.rs_folds,
        cv_test_size=args.cv_test_size,
        cv_splits=args.cv_folds,
        rs_iters=args.rs_iters,
        cv_random_state=42,
        scorer='adj_rand')

    # percent_labels_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    percent_labels_range = [-1] * 5
    percent_links_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    percent_unlabeled_range = [0.1, 0.2, 0.3, 0.4, 0.5]

    # no labels all unlabeled config v2

    context = OrderedDict(
        rs_test_size=args.rs_test_size,
        rs_splits=args.rs_folds,
        cv_test_size=args.cv_test_size,
        cv_splits=args.cv_folds,
        rs_iters=args.rs_iters,
        cv_random_state=42,
        scorer='adj_rand')

    # percent_labels_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    percent_labels_range = [0.2] * 6
    percent_links_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    percent_unlabeled_range = [0.4] * 6

    # few labels config

    # context = OrderedDict(
    #     rs_test_size=args.rs_test_size,
    #     rs_splits=args.rs_folds,
    #     cv_test_size=args.cv_test_size,
    #     cv_splits=args.cv_folds,
    #     rs_iters=args.rs_iters,
    #     cv_random_state=42,
    #     scorer='accuracy')
    #
    # percent_labels_range = [-1] * 6
    # percent_links_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # percent_unlabeled_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # svm adj rand config

    # context = OrderedDict(
    #     rs_test_size=args.rs_test_size,
    #     rs_splits=args.rs_folds,
    #     cv_test_size=args.cv_test_size,
    #     cv_splits=args.cv_folds,
    #     rs_iters=args.rs_iters,
    #     cv_random_state=42,
    #     scorer='adj_rand')
    #
    # # percent_labels_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    # percent_labels_range = [0.1, 0.2]
    # percent_links_range = []
    # percent_unlabeled_range = []

    assert args.estimators_file is not None
    estimators_module = imp.load_source('estimators', args.estimators_file)
    estimators = estimators_module.estimators


    def task_generator(estimators):
        for ds_name, (X, y) in datasets.iteritems():
            if issparse(X):
                X = X.toarray()
            context['dataset'] = ds_name

            for estimator_tuple in estimators:
                context['estimator'] = estimator_tuple.name

                if isinstance(estimator_tuple.estimator, LinksClassifier):

                    # labels_and_links = itertools.izip(
                    #     percent_labels_range,
                    #     percent_links_range)
                    # percents = [(labels, links, unlabeled) for (labels, links), unlabeled in
                    #             itertools.product(labels_and_links, percent_unlabeled_range)]
                    percents = itertools.izip_longest(
                        percent_labels_range,
                        percent_links_range,
                        percent_unlabeled_range,
                        fillvalue=0.0)
                elif isinstance(estimator_tuple.estimator, LogisticRegressionPairwise):
                    percents = itertools.izip_longest(
                        percent_labels_range[1:],
                        percent_links_range[1:],
                        [],
                        fillvalue=0.0)
                elif isinstance(estimator_tuple.estimator, SVC):
                    percents = itertools.izip_longest(np.unique(percent_labels_range),
                                                      [],
                                                      [], fillvalue=0.0)
                else:
                    raise Exception(
                        "I don't know what to do with %s" % type(estimator_tuple.estimator))
                percents = list(percents)

                for p_labels, p_links, p_unlabeled in percents:

                    # if not validate_percents(X, y, p_labels, p_links, p_unlabeled, disjoint=1):
                    #     continue

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

    runner = Runner(task=task,
                    task_generator=task_generator(estimators),
                    cacher=cacher,
                    mapper=mapper)
    runner.run()
