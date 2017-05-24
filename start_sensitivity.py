# coding: utf-8

# In[66]:
import argparse
from time import sleep
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocess as mp

import os
import itertools
from collections import OrderedDict
from functools import partial
from scipy.sparse import issparse

from sklearn.datasets import load_svmlight_file, make_circles, make_moons
import sys

from sklearn.model_selection import ShuffleSplit

from links import LinksClassifier
from logit import LogisticRegressionPairwise, LogisticRegression

from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm as tqdm

from new_experiment_runner.cacher import CSVCacher

# In[60]:

datafiles_toy = [
    r'data/diabetes_scale.libsvm',
    # r'data/breast-cancer_scale.libsvm',
]


# In[61]:

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


# In[79]:

datasets = OrderedDict([(os.path.split(f)[-1].replace('.libsvm', ''),
                         load_svmlight_file(f))
                        for f in datafiles_toy])

# In[80]:

datasets['circles'] = make_circles(n_samples=400, noise=0.1, factor=0.51)
datasets['moons'] = make_moons(n_samples=400, noise=0.1)


# In[82]:



# In[87]:

def split_dataset(X, y, percent_labels, percent_links, percent_unlabeled, random_state=42,
                  labels_and_links_separation_degree=0, return_index=False):
    """
    This function splits a dataset into 3 portions:
    1. labeled data
    2. links
    3. unlabeled data
    :param X:
    :param y:
    :param percent_labels:
    :param percent_links:
    :param percent_unlabeled:
    :param random_state:
    :param labels_and_links_separation_degree: 0 - random, 1 - at least 1 point is unlabeled, 2 - both unlabeled
    :param return_index
    :return: X(labeled), y(labels),
        X1(first point in link), X2(second point in link), z(must-link or cannot-link),
        Xu(unlabeled)
    """
    if random_state:
        np.random.seed(random_state)
    if issparse(X):
        X = X.toarray()

    if percent_links > 0:
        choice1 = next(StratifiedShuffleSplit(n_splits=1, train_size=percent_links).split(X, y))[0]
        choice1 = np.in1d(np.arange(len(y)), choice1)

        choice2 = next(StratifiedShuffleSplit(n_splits=1, train_size=percent_links).split(X, y))[0]
        choice2 = np.in1d(np.arange(len(y)), choice2)

        z = (y[choice1] == y[choice2]).astype(float)

        links_index = choice1 | choice2
    else:
        choice1 = np.zeros(len(y), dtype=bool)
        choice2 = np.zeros(len(y), dtype=bool)
        z = np.array([])
        links_index = choice1 | choice2
    # print(links_index.sum())


    if percent_labels < 1:
        if labels_and_links_separation_degree == 2:
            labels_where = np.where(~links_index)[0]
        elif labels_and_links_separation_degree == 1:
            labels_where = np.where(~ (choice1 & choice2))[0]
        else:
            labels_where = np.arange(len(y))
        labels_choice = next(StratifiedShuffleSplit(n_splits=1,
                                                    train_size=int(percent_labels * len(y))).split(
            X[labels_where], y[labels_where]))[0]

        # print(not_links_where.shape)
        labels_choice = labels_where[labels_choice]
    else:
        raise Exception()
        # labels_choice = np.arange(0, len(X))
    labels_index = np.in1d(np.arange(len(y)), labels_choice)

    unsup_index = ~(labels_index & links_index)
    unsup_where = np.where(unsup_index)[0]
    unsup_choice = np.random.choice(unsup_where, size=int(percent_unlabeled * len(y)),
                                    replace=False)

    # print(labels_index.sum(), links_index.sum(), unsup_index.sum())
    if labels_and_links_separation_degree == 2:
        assert (labels_index | links_index | unsup_index).sum() == \
               len(y) * (percent_labels + percent_links + percent_unlabeled)

    # assert labels_index.sum() == len(y) * percent_labels
    # assert links_index.sum() == len(y) * percent_links #TODO
    # assert unsup_index.sum() == len(y) * percent_unlabeled

    if return_index:
        return labels_index, choice1, choice2, z, unsup_choice
    return X[labels_index], y[labels_index], X[choice1], X[choice2], z, X[unsup_choice]


def split_dataset_stable(X, y, percent_labels, percent_links, percent_unlabeled, random_state=42,
                         labels_and_links_separation_degree=0, return_index=False):
    """
    This function splits a dataset into 3 portions:
    1. labeled data
    2. links
    3. unlabeled data
    It does so in a "stable" way, so that e.g. 15% percent of links are included in 30% of links,
        i.e. the same 15% plus another 15%

    :param X:
    :param y:
    :param percent_labels:
    :param percent_links:
    :param percent_unlabeled:
    :param random_state:
    :param labels_and_links_separation_degree: 0 - random, 1 - at least 1 point is unlabeled, 2 - both unlabeled
    :param return_index:
    :return: X(labeled), y(labels),
        X1(first point in link), X2(second point in link), z(must-link or cannot-link),
        Xu(unlabeled)
    """
    max_percent_labels, max_percent_links, max_percent_unlabeled = \
        get_max_percents(y, disjoint_labels_links=labels_and_links_separation_degree)

    assert percent_labels <= max_percent_labels \
           and percent_links <= max_percent_links \
           and percent_unlabeled <= max_percent_unlabeled

    labels, links1, links2, z, unsup = \
        split_dataset(X, y,
                      percent_labels=max_percent_labels,
                      percent_links=max_percent_links,
                      percent_unlabeled=max_percent_unlabeled,
                      random_state=random_state,
                      labels_and_links_separation_degree=labels_and_links_separation_degree,
                      return_index=True)

    labels = np.where(labels)[0]
    links1 = np.where(links1)[0]
    links2 = np.where(links2)[0]
    unsup = np.where(unsup)[0]

    if percent_labels == -1:
        labels_choice = np.arange(len(labels))
    elif percent_labels == 0:
        labels_choice = []
    else:
        _, labels_choice = next(StratifiedShuffleSplit(n_splits=1,
                                                       test_size=percent_labels,
                                                       random_state=42).split(
            np.zeros(shape=(len(labels), 0)),
            y[labels]))

    labels = labels[labels_choice]

    if percent_links == 0:
        links_choice = []
    else:
        _, links_choice = next(StratifiedShuffleSplit(n_splits=1,
                                                      test_size=percent_links,
                                                      random_state=42).split(
            np.zeros(shape=(len(z), 0)),
            z))
    links1 = links1[links_choice]
    links2 = links2[links_choice]
    z = z[links_choice]

    if percent_unlabeled == 0:
        unsup_choice = []
    else:
        _, unsup_choice = next(ShuffleSplit(n_splits=1,
                                            test_size=percent_unlabeled,
                                            random_state=42).split(
            np.zeros(shape=(len(unsup), 0)),
            unsup))

    unsup = unsup[unsup_choice]
    if return_index:
        return labels, links1, links2, z, unsup
    return X[labels], y[labels], X[links1], X[links2], z, X[unsup]


def get_max_percents(y, disjoint_labels_links):
    """
    Function to calculate maximum percent of
    labels, links and unlabeled data points for given class stratification
    Currently a heuristic
    :param y:
    :param disjoint_labels_links:
    :return: percent_labels, percent_links, percent_unlabeled
    """
    if disjoint_labels_links:
        return 0.4, 0.4, 0.4
    else:
        return 0.5, 0.5, 0.5


def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


def train_and_score(X_r, y_r, X1, X2, z, Xu, method, n_jobs=1):
    try:
        estimator = LinksClassifier(init=method, sampling='predefined',
                                    verbose=False, delta=0.01, beta=0.5,
                                    solver='tnc')

        grid = {
            # 'alpha': [0.01, 0.1, 1, 10],
            'gamma': [0.01, 0.05, 0.1, 0.3, 0.5, 1],
            'kernel': ['rbf'],
            # 'beta': [0.1, 0.2, 0.3, 0],
            # 'delta': []
        }
        full_index = np.ones(len(X_r), dtype=bool)
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
                          n_jobs=n_jobs,
                          verbose=False)
        gs.fit(X_r, y_r)
        last_loss = gs.best_estimator_.last_loss
        train_score = accuracy_scorer(gs, X_r, y_r)
    except Exception as e:
        e_t, e_v, e_tb = sys.exc_info()
        e_tb = traceback.format_tb(e_tb)
        return (e_t, e_v, e_tb)
    return last_loss, train_score


def call_wrapper(dataset, context):
    from start_sensitivity import train_and_score
    X_r, y_r, X1, X2, z, Xu = dataset
    result = train_and_score(X_r, y_r, X1, X2, z, Xu, context['method'])
    if isinstance(result[1], Exception):
        return context, result
    return context, {'loss': result[0], 'train_score': result[1]}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model evaluation script')

    parser.add_argument('--jobs', type=int, default=1,
                        help='number of parallel jobs, -1 for all')

    parser.add_argument('--file', type=str, default='data/results_sens.csv',
                        help='file to store results')

    parser.add_argument('--k', type=int, default=50,
                        help='number of tests')

    args = parser.parse_args()

    max_k = args.k
    cacher = CSVCacher(filename=args.file)


    def callback(context_result):
        context, result = context_result
        if isinstance(result[1], Exception):
            print(result[1])
            print('\n'.join(result[2]))
        else:
            cacher.set(context, result)
            cacher.save()


    n_jobs = args.jobs
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    # pool = mp.Pool(processes=n_jobs)
    # mp.freeze_support()
    results = []

    context = {}
    for ds_name, (X, y) in datasets.iteritems():
        context['dataset'] = ds_name
        print(ds_name)
        dataset_tuple = split_dataset(X, y, percent_labels=0.15, percent_links=0.15,
                                      percent_unlabeled=0.2)
        for method in ['zeros',
                       'normal',
                       'normal_univariate',
                       'normal_multivariate',
                       'random_labels',
                       'random_links_diff']:
            context['method'] = method
            print(method)
            for k in tqdm(range(max_k)):
                context['k'] = k

                if len(cacher.get(context) > 0):
                    continue

                last_loss, train_score = train_and_score(*dataset_tuple, method=method,
                                                         n_jobs=args.jobs)
                cacher.set(context, {'loss': last_loss, 'train_score': train_score})
                cacher.save()
                continue

                # res = pool.apply_async(call_wrapper,
                #                        kwds={
                #                            'dataset': dataset_tuple,
                #                            'context': context,
                #                        },
                #                        callback=callback)
                # results.append((context, res))

    tq = tqdm(total=len(results))

    while True:
        ready = 0
        for cntx, res in results:
            if res.ready():
                ready += 1
                callback(res.get())
        if ready == len(results):
            break
        if ready != tq.n:
            tq.update(ready - tq.n)
            # cacher.save()
        sleep(3)
        continue





# In[ ]:
