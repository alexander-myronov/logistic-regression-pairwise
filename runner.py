# -*- coding: utf-8 -*-
# In[ ]:
import argparse
import cProfile
from collections import OrderedDict
from io import StringIO
import json
import os
import pstats
import traceback
import datetime
from ipyparallel import CompositeError
import ipyparallel

import numpy as np
import itertools
import pandas as pd
from scipy.sparse import csr_matrix, issparse
import scipy
from six import BytesIO
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import time
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, roc_auc_score

import pathos.multiprocessing as mp

# In[ ]:



# In[ ]:

import sys
#
from experiment_runner.caching import MultipleFilesCacher
from experiment_runner.parallel_grid_search import GridSearchCVParallel
from experiment_runner.tqdm_callback import TqdmCallback
from logit import LogisticRegressionPairwise, LogisticRegression
from links import LinksClassifier

datafiles = [
    r'data/5ht2a_ExtFP.libsvm',
    r'data/5ht2c_ExtFP.libsvm',
    r'data/5ht6_ExtFP.libsvm',
    r'data/5ht7_ExtFP.libsvm',
    r'data/M1_ExtFP.libsvm',
    r'data/SERT_ExtFP.libsvm',
    r'data/cathepsin_ExtFP.libsvm',
    r'data/d2_ExtFP.libsvm',
    r'data/h1_ExtFP.libsvm',
    r'data/hERG_ExtFP.libsvm',
    r'data/hiv_integrase_ExtFP.libsvm',
    r'data/hiv_protease_ExtFP.libsvm',
]

datafiles_toy = [
    r'data/diabetes_scale.libsvm',
    r'data/australian_scale.libsvm',
    r'data/breast-cancer_scale.libsvm',
    r'data/german.numer_scale.libsvm',
    r'data/ionosphere_scale.libsvm',
    r'data/liver-disorders_scale.libsvm',
    r'data/heart_scale.libsvm',
]


# In[ ]:

def process_cm(confusion_mat, i=0):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i, i]  # correctly labeled as i
    FP = confusion_mat[:, i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i, :].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN
    return TP, FP, FN, TN


# In[ ]:

def bac_error(Y, Y_predict):
    import numpy as np
    cm = confusion_matrix(Y, Y_predict)
    bac_values = np.zeros(cm.shape[0])
    for i in xrange(cm.shape[0]):
        tp, fp, fn, tn = process_cm(cm, i=i)
        if tp + fn > 0 and tn + fp > 0:
            bac_values[i] = 0.5 * tp / (tp + fn) + 0.5 * tn / (tn + fp)
    return bac_values


# In[ ]:

def bac_scorer(estimator, X, Y):
    Y_predict = estimator.predict(X)
    bac_values = bac_error(Y, Y_predict)
    return np.mean(bac_values)


# In[ ]:

def perform_grid_search(estimator, X, y, scorer, param_grid, n_outer_folds,
                        n_outer_test_size,
                        n_inner_folds,
                        n_inner_test_size,
                        base_folder):
    """
    returns
    test score for each outer fold
    best score from grid search
    best estimator parameters for each iteration
    """
    test_scores = np.zeros(n_outer_folds)
    train_scores = np.zeros(n_outer_folds)
    best_parameters = []

    if issparse(X):
        X = X.toarray()

    folds = StratifiedShuffleSplit(n_splits=n_outer_folds,
                                   test_size=n_outer_test_size,
                                   random_state=42).split(X=X, y=y)

    mapper = mp.Pool(processes=4).imap_unordered
    # mapper = itertools.imap

    for n_fold, (train_index, test_index) in enumerate(folds):
        print('%d/%d fold' % (n_fold + 1, n_outer_folds))

        cache_dir = base_folder + '/' + 'fold=%d' % n_fold
        cacher = MultipleFilesCacher(cache_dir, flush_every_n=1)

        def record_metadata(index, fit_arguments):
            from experiment_runner.caching import MultipleFilesCacher
            meta_cacher = MultipleFilesCacher(cache_dir, flush_every_n=1,
                                              file_name_source=lambda
                                                  key: '%d_meta.pkl' % key)
            X = fit_arguments.pop('X')
            y = fit_arguments.pop('y')
            estimator = fit_arguments.pop('estimator')
            test = fit_arguments['test']
            y_pred = estimator.predict_proba(X[test])
            fit_arguments['y_pred_proba'] = y_pred
            fit_arguments['y_true'] = y[test]
            meta_cacher[index] = fit_arguments

        callback = TqdmCallback()
        search = GridSearchCVParallel(estimator, param_grid, scoring=scorer,
                                      cv=list(StratifiedShuffleSplit(n_splits=n_inner_folds,
                                                                     test_size=n_inner_test_size,
                                                                     random_state=43). \
                                              split(X[train_index], y[train_index])),
                                      verbose=2,
                                      refit=False,
                                      iid=False,
                                      mapper=mapper,
                                      callback=callback,
                                      cacher=cacher,
                                      fit_callback=record_metadata)

        search.fit(X[train_index], y[train_index])




        # x_is_index=True,
        # X_name=X_name,
        # y_name=y_name)

        best_estimator = clone(estimator).set_params(**search.best_params_)
        # best_estimator = search.best_estimator_
        best_estimator.fit(X[train_index], y[train_index])

        test_score = scorer(best_estimator, X[test_index], y[test_index])
        test_scores[n_fold] = test_score
        train_scores[n_fold] = search.best_score_

        best_parameters.append(search.best_params_)
        #             print('train score=%f, test score=%f' % (search.best_score_, test_score))
        print(search.best_params_)
    return test_scores, train_scores, best_parameters


# In[ ]:

def get_estimator_descritpion(estimator):
    name = type(estimator).__name__
    return name


# In[ ]:

def test_models(estimators, estimator_grids, X, Y, scorer, n_outer_folds, n_inner_folds,
                n_outer_test_size, n_inner_test_size, base_dir):
    estimator_scores = []
    # estimator_scores_std = np.zeros(len(estimators))
    assert len(estimators) <= len(estimator_grids)
    for i, (estimator, grid) in enumerate(itertools.izip(estimators, estimator_grids)):
        name = get_estimator_descritpion(estimator)

        scores_test, _, _ = perform_grid_search(estimator,
                                                X,
                                                Y,
                                                param_grid=grid,
                                                scorer=scorer,
                                                n_outer_folds=n_outer_folds,
                                                n_inner_folds=n_inner_folds,
                                                n_inner_test_size=n_inner_test_size,
                                                n_outer_test_size=n_outer_test_size,
                                                base_folder='%s/%s' % (base_dir, name))
        print(scores_test)
        estimator_scores.append(scores_test)

    return estimator_scores


# In[ ]:

# from twelm_theano import XELMTheano, EEMTheano, RBFNetTheano


estimator_grids_simple = [
    {'C': [None], 'h': [50]},
    {'C': [None], 'h': [50]},
    {'C': [None], 'h': [50]},
    {'C': [None], 'h': [50]},
    {'C': [1000], 'h': [50], 'b': [0.4]},
    {'C': [1000], 'h': [50]},
    {'C': [1000], 'h': [50]},
    {'C': [1000], 'h': [50]},
    {'C': [1000], 'h': [50]},
    {'n_estimators': [7, 12]}

]

estimators_toy = [
    # XELMTheano(f='euclidean', balanced=True),
    # EEMTheano(f='euclidean'),
    LogisticRegression(alpha=1),
    LogisticRegressionPairwise(),
    LinksClassifier(alpha=1, sampling='max_kdist'),
    # SVC(),
]

estimator_grids_toy = [
    {
        'kernel': ['rbf', ],
        'alpha': [0.1, 1, 10, 100, 1000, 10000],
        'gamma': ['auto', 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    },
    {
        'kernel': ['rbf', ],
        'alpha': [0.1, 1, 10, 100, 1000, 10000],
        'gamma': ['auto', 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        'mu': [0.01, 0.1, 1, 10, 100],
        'percent_pairs': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    },
    {
        'kernel': ['rbf', ],
        'alpha': [0.1, 1, 10, 100, 1000, 10000],
        'kernel_gamma': ['auto', 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        'gamma': [0.01, 0.1, 1, 10, 100],
        'percent_pairs': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    },
    # {
    #     'kernel': ['rbf'],
    #     'C': [0.1, 1, 10, 100, 1000, 10000],
    #     'gamma': ['auto', 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    # },

]


def roc_prob_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y_pred = estimator.predict_proba(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return roc_auc_score(y_true, y_pred[:, 1])


def accuracy_scorer(estimator, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X)
    y_true = np.copy(y)
    y_true[y_true == -1] = 0
    return accuracy_score(y_true, y_pred)


def prepare_and_train(name, X, y, estimator_index, n_inner_folds=3, n_inner_test_size=0.2,
                      n_outer_folds=3, n_outer_test_size=0.2):
    print(name, get_estimator_descritpion(estimators[estimator_index]))

    try:
        # pr = cProfile.Profile()
        # pr.enable()

        estimators_scores = test_models([estimators[estimator_index]],
                                        [estimator_grids[estimator_index]],
                                        X,
                                        y,
                                        scorer=accuracy_scorer,
                                        n_outer_folds=n_outer_folds,
                                        n_outer_test_size=n_outer_test_size,
                                        n_inner_folds=n_inner_folds,
                                        n_inner_test_size=n_inner_test_size,
                                        base_dir='data/%s' % name)
        # pr.disable()

        # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        # ps.print_stats()
    except Exception:
        e_type, e_value, e_tb = sys.exc_info()
        # print(e_tb)

        return name, estimator_index, (e_value, e_tb)
    print('%s ready' % name)
    return name, estimator_index, estimators_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='cross validation number of folds')
    parser.add_argument('--cv_test_size', type=float, default=0.2,
                        help='cross validation test size')
    parser.add_argument('--gs_folds', type=int, default=3,
                        help='grid search number of folds')
    parser.add_argument('--gs_test_size', type=float, default=0.2,
                        help='grid search test size')
    parser.add_argument('--file', type=str,
                        default=r'data/results%s.csv' % \
                                str(datetime.datetime.now()).replace(':', '.'),
                        help='results file name')

    args = parser.parse_args()

    cache = True

    datasets = OrderedDict([(os.path.split(f)[-1].replace('.libsvm', ''), load_svmlight_file(f))
                            for f in datafiles_toy])




    # exit()

    estimators = estimators_toy
    estimator_grids = estimator_grids_toy

    filename = r'data/results_debug.csv'
    filename = args.file

    if os.path.isfile(filename):
        scores_grid = pd.read_csv(filename)
        scores_grid.loc[:, 'dataset'] = pd.Series(data=datasets.keys())
    else:
        scores_grid = pd.DataFrame(dtype=object)
        scores_grid.loc[:, 'dataset'] = pd.Series(data=datasets.keys())

    scores_grid.set_index('dataset', inplace=True, drop=False)
    scores_grid = scores_grid.reindex(pd.Series(data=datasets.keys()))

    for estimator in estimators:
        column_name = get_estimator_descritpion(estimator)
        if column_name not in scores_grid.columns:
            scores_grid.loc[:, column_name] = pd.Series(
                [''] * len(datasets)).astype('str')
        scores_grid.loc[:, column_name] = scores_grid.loc[:, column_name].astype('str')


    def map_callback(result):
        dataset_name, estimator_index, estimators_scores = result
        estimator_name = get_estimator_descritpion(estimators[estimator_index])
        print('callback %s dataset %s model' % (dataset_name, estimator_name))
        #  print(result[2])

        if isinstance(estimators_scores[0], Exception):

            tb = ''.join(traceback.format_tb(estimators_scores[1]))
            print(
                'callback: datafile=%s, model=%s exception=%s \n traceback\n %s' % (
                    dataset_name, estimator_name, repr(estimators_scores[0]), tb))
            if isinstance(estimators_scores[0], CompositeError):
                estimators_scores[0].print_traceback()
        else:
            scores_grid.set_value(dataset_name, estimator_name, str(estimators_scores[0]))
            scores_grid.to_csv(filename, index=False)


    start_time = time.time()

    parallel_models = False
    if parallel_models:

        import pathos.multiprocessing as mp

        pool = mp.Pool(processes=4)

        async_results = []
        for (name, (dataset)), estimator_index in itertools.product(datasets.items()[::1],
                                                                    xrange(0, len(estimators))):
            value = \
                str(scores_grid.loc[name, get_estimator_descritpion(estimators[estimator_index])])

            try:
                value = np.fromstring(value.replace('[', '').replace(']', ''), sep=' ')
                if np.isnan(value).any():
                    raise Exception
            except:
                value = None

            if callable(dataset):
                X, y = dataset()
            else:
                X, y = dataset

            if value is None or len(value) < args.cv_folds:
                res = pool.apply_async(prepare_and_train,
                                       args=(name,
                                             X,
                                             y,
                                             estimator_index,
                                             ),
                                       kwds={
                                           'n_inner_folds': args.gs_folds,
                                           'n_inner_test_size': args.gs_test_size,
                                           'n_outer_folds': args.cv_folds,
                                           'n_outer_test_size': args.cv_reps
                                       },
                                       callback=map_callback)
                async_results.append(res)
        while not all(map(lambda r: r.ready(), async_results)):
            pass
        # result_series = map(lambda ar: ar.get(), async_results)
        pool.close()
    else:

        #

        for (name, (dataset)), estimator_index in itertools.product(datasets.items()[::1],
                                                                    xrange(0, len(estimators))):
            value = \
                str(scores_grid.loc[name, get_estimator_descritpion(estimators[estimator_index])])

            try:
                value = np.fromstring(value.replace('[', '').replace(']', ''), sep=' ')
                if np.isnan(value).any():
                    raise Exception
            except:
                value = None

            if callable(dataset):
                X, y = dataset()
            else:
                X, y = dataset

            if value is None or len(value) < args.cv_folds:
                map_callback(prepare_and_train(name, X, y, estimator_index,
                                               n_inner_folds=args.gs_folds,
                                               n_inner_test_size=args.gs_test_size,
                                               n_outer_folds=args.cv_folds,
                                               n_outer_test_size=args.cv_test_size,
                                               ))

    end_time = time.time()

    print 'training done in %2.2f sec' % (end_time - start_time)

    exit()
