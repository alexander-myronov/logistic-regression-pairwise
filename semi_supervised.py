from label_link_tradeoff import plot_tradeoff

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

from sklearn.datasets import load_svmlight_file

from links import LinksClassifier
from logit import LogisticRegressionPairwise, LogisticRegression

from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit, GridSearchCV

from tqdm import tqdm as tqdm


def sample_links_random(X, y, percent_links):
    # np.random.seed(44)
    num = int(len(y) * percent_links)

    choice1 = np.random.choice(len(y), size=num, replace=True)
    X1 = X[choice1]
    choice2 = np.random.choice(len(y), size=num, replace=True)
    X2 = X[choice2]
    z = (y[choice1] == y[choice2]).astype(float)

    return X1, X2, z


def split_dataset(X, y, percent_labels, percent_links):
    X1, X2, z = sample_links_random(X, y, percent_links)

    if percent_labels < 1:
        labels_choice = \
            next(StratifiedShuffleSplit(n_splits=1, train_size=percent_labels).split(X, y))[0]
    else:
        labels_choice = np.arange(0, len(X))
    index = np.zeros(len(X), dtype=bool)
    index[labels_choice] = True
    return X[index], y[index], X1, X2, z, X[~index]


links_grid = {
    'alpha': [0.01, 0.1, 1, 10],
    'kernel_gamma': ['auto'],
    'kernel': ['rbf'],
    'gamma': [0.1, 0.1, 1, 10]
}


def train_labels(X, y, X1, X2, z, Xu, n_jobs=1):
    estimator = LinksClassifier()
    grid = {
        'alpha': [0.01, 0.1, 1, 10],
        'kernel_gamma': ['auto'],
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


def train_labels_links(X, y, X1, X2, z, Xu, n_jobs=1):
    estimator = LinksClassifier()
    grid = {
        'alpha': [0.01, 0.1, 1, 10],
        'kernel_gamma': ['auto'],
        'kernel': ['rbf'],
        'gamma': [0.1, 0.1, 1, 10]
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


def train_labels_links_unlabeled(X, y, X1, X2, z, Xu, n_jobs=1):
    estimator = LinksClassifier()
    grid = {
        'alpha': [0.01, 0.1, 1, 10],
        'kernel_gamma': ['auto'],
        'kernel': ['rbf'],
        'gamma': [0.1, 0.1, 1, 10],
        'delta': [0.1, 0.1, 1, 10]
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
                      verbose=2,
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
        r'data/australian_scale.libsvm',
        r'data/breast-cancer_scale.libsvm',
        r'data/german.numer_scale.libsvm',
        r'data/ionosphere_scale.libsvm',
        r'data/liver-disorders_scale.libsvm',
        r'data/heart_scale.libsvm',
    ]


    # In[4]:

    def loader(name):
        from sklearn.datasets import load_svmlight_file
        from scipy.sparse import issparse
        filename = 'data/%s.libsvm' % name
        if not name in globals():
            X, y = load_svmlight_file(filename)
            if issparse(X):
                X = X.toarray()
            globals()[name] = (X, y)
        return globals()[name]


    # In[5]:

    datasets = OrderedDict([(os.path.split(f)[-1].replace('.libsvm', ''),
                             partial(loader, os.path.split(f)[-1].replace('.libsvm', '')))
                            for f in datafiles_toy])

    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='cross validation number of folds')
    parser.add_argument('--cv_test_size', type=float, default=0.2,
                        help='cross validation test size')

    parser.add_argument('--jobs', type=int, default=1,
                        help='number of parallel jobs, -1 for all')

    parser.add_argument('--dir', type=str, default='data/',
                        help='folder to store results')

    parser.add_argument('--plot', type=bool, default=False,
                        help='folder to store results')

    estimators = [
                     ('LinksClassifier-labels', train_labels),
                     ('LinksClassifier-labels-links', train_labels_links),
                     ('LinksClassifier-labels-links-unlabeled', train_labels_links_unlabeled),
                     ('LogisticRegression', train_labels_logit),

                 ][::-1]

    args = parser.parse_args()
    for ds_name, loader in datasets.iteritems():
        X, y = loader()
        print(ds_name)
        ds_dir = os.path.join(args.dir, ds_name)
        for est_name, estimator_train_f in estimators:
            print(est_name)
            est_dir = os.path.join(ds_dir, est_name)
            if not os.path.exists(est_dir):
                os.makedirs(est_dir)
            percent_labels_range = np.linspace(0.1, 1, 3)
            percent_links_range = np.linspace(0.1, 1, 3)
            outer_cv = list(
                StratifiedShuffleSplit(n_splits=args.cv_folds,
                                       test_size=args.cv_test_size,
                                       random_state=42).split(X, y))

            train_scores_filename = os.path.join(est_dir, 'train_scores.npy')
            test_scores_filename = os.path.join(est_dir, 'test_scores.npy')

            if os.path.isfile(train_scores_filename):
                train_scores = np.load(train_scores_filename)
            else:
                train_scores = np.full(
                    shape=(len(percent_labels_range), len(percent_links_range), len(outer_cv)),
                    fill_value=-1, dtype=float)
            if os.path.isfile(train_scores_filename):
                test_scores = np.load(train_scores_filename)
            else:
                test_scores = np.full(
                    shape=(len(percent_labels_range), len(percent_links_range), len(outer_cv)),
                    fill_value=-1, dtype=float)

            if args.plot:
                fig = plot_tradeoff(train_scores, test_scores,
                                    range_x=percent_labels_range,
                                    range_y=percent_links_range)
                fig.set_size_inches(15, 10)
                fig.savefig(os.path.join(est_dir, 'tradeoff.png'), bbox_inches='tight', dpi=300)

            #tq = tqdm(total=len(percent_links_range) * len(percent_labels_range), desc='')

            for (i_label, p_labels), (i_link, p_links) in \
                    itertools.product(enumerate(percent_labels_range),
                                      enumerate(percent_links_range)):
                #tq.set_description('labels=%.2f, links=%.2f' % (p_labels, p_links))
                #tq.update(1)

                for i_split, (train, test) in enumerate(outer_cv):
                    if train_scores[i_label, i_link, i_split] != -1 and \
                                    test_scores[i_label, i_link, i_split] != -1:
                        continue
                    X_train, y_train, X1_train, X2_train, z_train, Xu_train = \
                        split_dataset(X[train],
                                      y[train],
                                      percent_labels=p_labels,
                                      percent_links=p_links)

                    gs = estimator_train_f(X_train, y_train, X1_train, X2_train, z_train, Xu_train,
                                           n_jobs=args.jobs)

                    tr_score = gs.best_score_
                    # print('tr score', tr_score)
                    train_scores[i_label, i_link, i_split] = tr_score

                    te_score = accuracy_scorer(gs, X[test], y[test])
                    # print('te score', te_score)
                    test_scores[i_label, i_link, i_split] = te_score

                    np.save(train_scores_filename, train_scores)
                    np.save(test_scores_filename, test_scores)

                    if args.plot:
                        fig = plot_tradeoff(train_scores, test_scores,
                                            range_x=percent_labels_range,
                                            range_y=percent_links_range)
                        fig.set_size_inches(15, 10)
                        fig.savefig(os.path.join(est_dir, 'tradeoff.png'), bbox_inches='tight',
                                    dpi=300)
                        tq = tqdm(total=len(percent_links_range) * len(percent_labels_range),
                                  desc='')
