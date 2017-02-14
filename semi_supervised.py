from scipy.sparse import issparse
from label_link_tradeoff import plot_tradeoff
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

from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit, GridSearchCV

from tqdm import tqdm as tqdm

from start_sensitivity import split_dataset

links_grid = {
    'alpha': [0.01, 0.1, 1, 10],
    'gamma': ['auto'],
    'kernel': ['rbf'],
    'beta': [0.1, 0.1, 1, 10]
}


def train_labels(X, y, X1, X2, z, Xu, n_jobs=1):
    estimator = LinksClassifier(sampling='predefined', init='normal_univariate')
    grid = {
        'alpha': [0.01, 0.1, 1, 10],
        'gamma': ['auto'],
        'kernel': ['rbf'],
        # 'gamma': [0.1, 0.1, 1, 10]
    }
    full_index = np.ones(len(X), dtype=bool)

    gs = GridSearchCV(estimator=estimator,
                      param_grid=grid,
                      cv=[(full_index, full_index)],
                      scoring=accuracy_scorer,
                      fit_params={
                      },
                      refit=True,
                      n_jobs=n_jobs)
    gs.fit(X, y)
    return gs


def train_labels_links(X, y, X1, X2, z, Xu, n_jobs=1, kernel='rbf'):
    estimator = LinksClassifier(sampling='predefined', init='zeros', verbose=False, solver='tnc')
    grid = {
        #'alpha': [0.01, 0.1, 1, 10],
        'gamma': [0.01, 0.05, 0.1, 0.5, 1],
        'kernel': [kernel],
        # 'delta': [0.1, 0.1, 1, 10]
    }
    full_index = np.ones(len(X), dtype=bool)

    gs = GridSearchCV(estimator=estimator,
                      param_grid=grid,
                      cv=[(full_index, full_index)],
                      scoring=accuracy_scorer,
                      fit_params={
                          'X1': X1,
                          'X2': X2,
                          'z': z,
                          'Xu': np.zeros(shape=(0, X.shape[1]))
                      },
                      refit=True,
                      n_jobs=n_jobs)
    gs.fit(X, y)
    return gs


def train_labels_links_unlabeled(X, y, X1, X2, z, Xu, n_jobs=1, kernel='rbf'):
    estimator = LinksClassifier(sampling='predefined', init='zeros', verbose=False, solver='tnc')
    grid = {
        #'alpha': [0.01, 0.1, 1, 10],
        'gamma': [0.01, 0.05, 0.1, 0.5, 1],
        'kernel': [kernel],
        'delta': [0.01, 0.1, 0.3, 0.6]
    }
    full_index = np.ones(len(X), dtype=bool)

    gs = GridSearchCV(estimator=estimator,
                      param_grid=grid,
                      cv=[(full_index, full_index)],
                      scoring=accuracy_scorer,
                      fit_params={
                          'X1': X1,
                          'X2': X2,
                          'z': z,
                          'Xu': Xu
                      },
                      refit=True,
                      verbose=0,
                      n_jobs=n_jobs)
    gs.fit(X, y)
    return gs


def train_labels_logit(X, y, X1, X2, z, Xu, n_jobs=1):
    estimator = LogisticRegression(alpha=1)
    grid = {
        'kernel': ['rbf', ],
        'alpha': [0.1, 1, 10],
        'gamma': ['auto']
    }
    full_index = np.ones(len(X), dtype=bool)

    gs = GridSearchCV(estimator=estimator,
                      param_grid=grid,
                      cv=[(full_index, full_index)],
                      scoring=accuracy_scorer,
                      fit_params={
                      },
                      refit=True,
                      n_jobs=n_jobs)
    gs.fit(X, y)
    return gs


def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


if __name__ == '__main__':

    # In[3]:

    datafiles_toy = [
        r'data/diabetes_scale.libsvm',
        r'data/breast-cancer_scale.libsvm',
        r'data/australian_scale.libsvm',
        # r'data/ionosphere_scale.libsvm',
    ]

    datasets = OrderedDict([(os.path.split(f)[-1].replace('.libsvm', ''),
                             load_svmlight_file(f))
                            for f in datafiles_toy])
    datasets['circles'] = make_circles(n_samples=400, noise=0.1, factor=0.51)
    datasets['moons'] = make_moons(n_samples=400, noise=0.1)

    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='cross validation number of folds')
    parser.add_argument('--cv_test_size', type=float, default=0.2,
                        help='cross validation test size')

    parser.add_argument('--jobs', type=int, default=1,
                        help='number of parallel jobs, -1 for all')

    parser.add_argument('--file', type=str, default='data/results_semi.csv',
                        help='folder to store results')

    estimators = [
        # ('LinksClassifier-labels', train_labels),
        ('LinksClassifier-labels-links', train_labels_links),
        ('LinksClassifier-labels-links-unlabeled', train_labels_links_unlabeled),
        # ('LogisticRegression', train_labels_logit),

    ]

    args = parser.parse_args()
    cacher = CSVCacher(filename=args.file)

    context = {'cv_test_size': args.cv_test_size}
    context = {'cv_random_state': 42}
    for ds_name, (X, y) in datasets.iteritems():
        if issparse(X):
            X = X.toarray()
        context['dataset'] = ds_name
        print(ds_name)
        for est_name, estimator_train_f in estimators:
            print(est_name)
            context['estimator'] = est_name

            percent_labels_range = [0.2]  # np.linspace(0.1, 0.3, 5)
            percent_links_range = [0.2]  # np.linspace(0.1, 0.3, 5)
            percent_unlabeled_range = [0.2]
            outer_cv = list(
                StratifiedShuffleSplit(n_splits=args.cv_folds,
                                       test_size=args.cv_test_size,
                                       random_state=42).split(X, y))

            for (i_label, p_labels), (i_link, p_links), (i_unlableled, p_unlabeled) in \
                    itertools.product(enumerate(percent_labels_range),
                                      enumerate(percent_links_range),
                                      enumerate(percent_unlabeled_range)):

                context['percent_labels'] = p_labels
                context['percent_links'] = p_links
                context['percent_unlabeled'] = p_unlabeled

                for i_split, (train, test) in tqdm(list(enumerate(outer_cv))):
                    context['cv_split'] = i_split
                    X_train, y_train, X1_train, X2_train, z_train, Xu_train = \
                        split_dataset(X[train],
                                      y[train],
                                      percent_labels=p_labels,
                                      percent_links=p_links,
                                      percent_unlabeled=p_unlabeled)
                    if len(cacher.get(context) > 1):
                        continue

                    gs = estimator_train_f(X_train, y_train, X1_train, X2_train, z_train, Xu_train,
                                           n_jobs=args.jobs, kernel='rbf')

                    tr_score = gs.best_score_
                    # print('tr score', tr_score)

                    te_score = accuracy_scorer(gs, X[test], y[test])
                    # print('te score', te_score)


                    cacher.set(context, {'train_score': tr_score, 'test_score': te_score})
                    cacher.save()
