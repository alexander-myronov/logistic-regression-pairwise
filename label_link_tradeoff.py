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

estimators = [
    ('LinksClassifier',
     LinksClassifier(sampling='predefined'),
     {
         'alpha': [0.01, 0.1, 1, 10],
         'kernel_gamma': ['auto'],
         'kernel': ['rbf'],
         'gamma': [0.1, 0.1, 1, 10]
     }),
]


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
    n_labels = int(len(y) * percent_labels)
    labels_choice = \
        next(StratifiedShuffleSplit(n_splits=1, train_size=percent_labels).split(X, y))[0]
    return X[labels_choice], y[labels_choice], X1, X2, z


def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


def plot_scores(ax, scores, vmin, vmax, range_x, range_y):
    r = ax.imshow(scores.mean(axis=2), interpolation='nearest',
                  cmap=plt.cm.hot,
                  vmax=1,
                  vmin=vmin, origin='lower')
    ax.set_xticks(np.arange(len(range_x)))
    ax.set_xticklabels(range_x)
    ax.set_yticks(np.arange(len(range_y)))
    ax.set_yticklabels(range_y)
    return r


def plot_tradeoff(train_scores, test_scores):
    fig, ax = plt.subplots(ncols=2)
    vmin = min(train_scores.min(), test_scores.min())
    r = plot_scores(ax[0], train_scores, vmin=vmin, vmax=1,
                    range_x=percent_labels_range,
                    range_y=percent_links_range)
    ax[0].set_title('train score')

    r = plot_scores(ax[1], test_scores, vmin=vmin, vmax=1,
                    range_x=percent_labels_range,
                    range_y=percent_links_range)
    ax[1].set_title('test score')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(r, cax=cbar_ax)
    return fig


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

    parser.add_argument('--jobs', type=int, default=0,
                        help='number of parallel jobs, -1 for all')

    parser.add_argument('--dir', type=str, default='data/',
                        help='folder to store results')

    args = parser.parse_args()
    for ds_name, loader in datasets.iteritems():
        X, y = loader()
        print(ds_name)
        ds_dir = os.path.join(args.dir, ds_name)
        for est_name, estimator, grid in estimators:
            print(est_name)
            est_dir = os.path.join(ds_dir, est_name)
            if not os.path.exists(est_dir):
                os.makedirs(est_dir)
            percent_labels_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            percent_links_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
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

            fig = plot_tradeoff(train_scores, test_scores)
            fig.set_size_inches(15, 10)

            fig.savefig(os.path.join(est_dir, 'tradeoff.png'), bbox_inches='tight', dpi=300)

            tq = tqdm(total=len(percent_links_range) * len(percent_labels_range), desc='')

            for (i_label, p_labels), (i_link, p_links) in \
                    itertools.product(enumerate(percent_labels_range),
                                      enumerate(percent_links_range)):
                tq.set_description('labels=%.2f, links=%.2f' % (p_labels, p_links))
                tq.update(1)

                for i_split, (train, test) in enumerate(outer_cv):
                    if train_scores[i_label, i_link, i_split] != -1 and \
                                    test_scores[i_label, i_link, i_split] != -1:
                        continue
                    X_train, y_train, X1_train, X2_train, z_train = split_dataset(X[train],
                                                                                  y[train],
                                                                                  percent_labels=p_labels,
                                                                                  percent_links=p_links)

                    full_index = np.ones(len(X_train), dtype=bool)

                    gs = GridSearchCV(estimator=estimator,
                                      param_grid=grid,
                                      cv=[(full_index, full_index)], scoring=accuracy_scorer,
                                      fit_params={
                                          'X1': X1_train,
                                          'X2': X2_train,
                                          'z': z_train,
                                          'Xu': np.zeros(shape=(0, X.shape[1]))},
                                      refit=True,
                                      n_jobs=-1)
                    gs.fit(X_train, y_train)
                    tr_score = gs.best_score_
                    # print('tr score', tr_score)
                    train_scores[i_label, i_link, i_split] = tr_score

                    te_score = accuracy_scorer(gs, X[test], y[test])
                    # print('te score', te_score)
                    test_scores[i_label, i_link, i_split] = te_score

                    np.save(train_scores_filename, train_scores)
                    np.save(test_scores_filename, test_scores)






# In[ ]:
