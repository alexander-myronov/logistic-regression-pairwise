# coding: utf-8

# In[66]:
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocess as mp

import os
import itertools
from collections import OrderedDict
from functools import partial
from scipy.sparse import issparse

from sklearn.datasets import load_svmlight_file, make_circles

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
    r'data/breast-cancer_scale.libsvm',
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

def split_dataset(X, y, percent_labels, percent_links, unlabeled=True, random_state=42):
    if random_state:
        np.random.seed(random_state)
    if issparse(X):
        X = X.toarray()
    choice1 = next(StratifiedShuffleSplit(n_splits=1, train_size=percent_links).split(X, y))[0]
    choice1 = np.in1d(np.arange(len(y)), choice1)

    choice2 = next(StratifiedShuffleSplit(n_splits=1, train_size=percent_links).split(X, y))[0]
    choice2 = np.in1d(np.arange(len(y)), choice2)

    z = (y[choice1] == y[choice2]).astype(float)

    links_index = choice1 | choice2
    # print(links_index.sum())


    if percent_labels < 1:
        not_links_where = np.where(~links_index)[0]
        labels_choice = next(StratifiedShuffleSplit(n_splits=1,
                                                    train_size=int(percent_labels * len(y))).split(
            X[not_links_where], y[not_links_where]))[0]

        # print(not_links_where.shape)
        labels_choice = not_links_where[labels_choice]
    else:
        raise Exception()
        # labels_choice = np.arange(0, len(X))
    labels_index = np.in1d(np.arange(len(y)), labels_choice)

    unsup_index = ~(labels_index & links_index)

    # print(labels_index.sum(), links_index.sum(), unsup_index.sum())
    assert (labels_index | links_index | unsup_index).sum() == len(y)

    return X[labels_index], y[labels_index], X[choice1], X[choice2], z, X[unsup_index]


# In[88]:

def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


def train_and_score(X_r, y_r, X1, X2, z, Xu, method, n_jobs=1):
    estimator = LinksClassifier(init=method, sampling='predefined',
                                verbose=False, delta=0.00, beta=0.5)
    grid = {
        # 'alpha': [0.01, 0.1, 1, 10],
        'gamma': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 2],
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
    return last_loss, train_score


def call_wrapper(dataset, context):
    from start_sensitivity import train_and_score
    X_r, y_r, X1, X2, z, Xu = dataset
    result = train_and_score(X_r, y_r, X1, X2, z, Xu, context['method'])
    return context, {'loss': result[0], 'train_score': result[1]}


if __name__ == '__main__':

    max_k = 50
    cacher = CSVCacher(filename='data/start_sensitivity.csv')


    def callback(context_result):
        context, result = context_result
        cacher.set(context, result)
        cacher.save()


    pool = mp.Pool(processes=30)
    mp.freeze_support()
    results = []

    context = {}
    for ds_name, (X, y) in datasets.iteritems():
        context['dataset'] = ds_name
        print(ds_name)
        dataset_tuple = split_dataset(X, y, percent_labels=0.3, percent_links=0.3,
                                      unlabeled=True)
        for method in ['normal_univariate',
                       'normal_multivariate',
                       'random_labels',
                       'random_links_diff']:
            context['method'] = method
            print(method)
            for k in tqdm(range(max_k)):
                context['k'] = k

                if len(cacher.get(context) > 0):
                    continue

                # last_loss, train_score = train_and_score(*dataset_tuple, method=method, n_jobs=1)
                # cacher.set(context, {'loss': last_loss, 'train_score': train_score})
                # cacher.save()

                res = pool.apply_async(call_wrapper,
                                       kwds={
                                           'dataset': dataset_tuple,
                                           'context': context,
                                       },
                                       callback=callback)
                results.append(res)

    tq = tqdm(total=len(results))

    while True:
        ready = sum(map(lambda r: r.ready(), results))
        if ready == len(results):
            break
        if ready != tq.n:
            tq.update(ready - tq.n)
        sleep(3)
        continue





# In[ ]:
