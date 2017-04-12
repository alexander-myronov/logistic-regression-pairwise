from __future__ import division, print_function

import glob
import traceback

from collections import defaultdict

import gc
from scipy.sparse import issparse
from sklearn import clone
import sys

from sklearn.model_selection import ParameterSampler

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
import scipy.io

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

from start_sensitivity import split_dataset_stable
from new_experiment_runner.cacher import CSVCacher
from links_vs_npklr_vs_svm import split_datasets, accuracy_scorer, validate_percents, \
    estimator_tuple, task
from sklearn.model_selection import ParameterSampler
from sklearn.base import clone


def load_matlab_context(filename, my_contexts):
    mat = scipy.io.loadmat(filename, struct_as_record=False)
    if 'stuff' not in mat:
        print(mat.keys())
        return []
    contexts = mat['stuff']
    results = []
    for mat_context, my_context in itertools.izip(contexts.ravel(), my_contexts):
        mat_context = mat_context[0, 0]

        # assert that:
        # 1. all required keys are scalar
        # 2. all key values are equal to the desired context

        def compare_values(mat_value, my_value):
            if mat_value.shape == (1, 1):
                mat_value = mat_value[0, 0]
            elif mat_value.shape == (1,):
                mat_value = mat_value[0]
            if isinstance(mat_value, str) or isinstance(my_value, str):
                return mat_value == my_value
            if type(mat_value) != type(my_value):
                return float(mat_value) == float(my_value)

        assert all([compare_values(mat_context.__getattribute__(key), value)
                    for key, value in my_context.iteritems()])

        result = {}
        result['test_score'] = mat_context.test_score[0, 0]
        result['cv_score'] = mat_context.cv_score[0, 0]

        result['gamma'] = mat_context.gamma[0, 0]
        result['gamma_I'] = mat_context.gamma_I[0, 0]
        result['gamma_A'] = mat_context.gamma_A[0, 0]
        results.append(result)

    return results


if __name__ == '__main__':

    # aa = scipy.io.loadmat('../lapsvmp_v02/lapsvm_stuff_ready.mat', struct_as_record=False)
    # aa = aa['stuff'][0, 0]
    # print(aa)
    # exit()
    # mp.freeze_support()

    datafiles_toy = [
        r'data/diabetes_scale.libsvm',
        r'data/breast-cancer_scale.libsvm',
        r'data/australian_scale.libsvm',
        r'data/ionosphere_scale.libsvm',
        # r'data/german.numer_scale.libsvm',
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

    parser.add_argument('--file', type=str, default='data/lapsvm.csv',
                        help='folder to store results')

    args = parser.parse_args()
    cacher = CSVCacher(filename=args.file)


    # context = {'rs_test_size': args.rs_test_size,
    #            'rs_splits': args.rs_folds,
    #            'cv_test_size': args.cv_test_size,
    #            'cv_splits': args.cv_folds,
    #            'rs_iters': args.rs_iters,
    #            'cv_random_state': 42}




    def task_generator(grid_func, percents, do_random_search=True):

        # contexts_by_ds = defaultdict(list)
        for ds_name, (X, y) in datasets.iteritems():
            ds_contexts = []
            if issparse(X):
                X = X.toarray()
            context = OrderedDict()
            context['dataset'] = ds_name

            # percents = itertools.product(percent_labels_range,
            #                              percent_links_range,
            #                              percent_unlabeled_range)


            for p_labels, p_links, p_unlabeled in percents:

                if not validate_percents(X, y, p_labels, p_links, p_unlabeled, disjoint=False):
                    continue

                context['percent_labels'] = p_labels
                context['percent_links'] = p_links
                context['percent_unlabeled'] = p_unlabeled

                outer_cv = StratifiedShuffleSplit(n_splits=args.cv_folds,
                                                  test_size=args.cv_test_size,
                                                  random_state=42).split(X, y)

                for i_split, (train, test) in enumerate(outer_cv):
                    context['cv_split'] = i_split

                    X_train, y_train = X[train], y[train]
                    X_tr, y_tr, X1_tr, X2_tr, z_tr, Xu_tr = split_dataset_stable(
                        X_train, y_train,
                        percent_labels=context['percent_labels'],
                        percent_links=context['percent_links'],
                        percent_unlabeled=context['percent_unlabeled'],
                        disjoint_labels_and_links=False,
                        random_state=42)
                    if do_random_search:
                        rs_contexts = []
                        for i_inner_split, ((tr_y, tr_z, tr_u), (te_y, te_z, te_u)) \
                                in enumerate(
                            split_datasets(
                                X_tr,
                                y_tr,
                                X1_tr,
                                X2_tr,
                                z_tr,
                                Xu_tr,
                                n_splits=args.rs_folds,
                                test_size=args.rs_test_size)):
                            rs_context = OrderedDict()
                            rs_context['rs_split'] = i_inner_split

                            fit_kwargs = {
                                'X1': X1_tr[tr_z],
                                'X2': X2_tr[tr_z],
                                'z': z_tr[tr_z],
                                'Xu': Xu_tr[tr_u],
                            }
                            grid = grid_func(X_tr, y_tr, fit_kwargs)

                            for params in ParameterSampler(grid, args.rs_iters, random_state=42):
                                rs_context.update(params)

                                rs_context.update(fit_kwargs)
                                rs_context['X'] = X_tr[tr_y]
                                rs_context['y'] = y_tr[tr_y]
                                rs_context['X_val'] = X_tr[te_y]
                                rs_context['y_val'] = y_tr[te_y]
                                rs_contexts.append(OrderedDict(rs_context))
                        context['random_search'] = rs_contexts

                    context['X_full'] = X
                    context['y_full'] = y
                    context['X'] = X_tr
                    context['y'] = y_tr
                    context['X1'] = X1_tr
                    context['X2'] = X2_tr
                    context['z'] = z_tr
                    context['Xu'] = Xu_tr

                    context['X_test'] = X[test]
                    context['y_test'] = y[test]

                    context['train'] = train
                    context['test'] = test
                    # contexts_by_ds[ds_name].append(dict(context))
                    ds_contexts.append(OrderedDict(context))
            yield ds_name, ds_contexts
            # return contexts_by_ds


    def clean_context(context):
        if isinstance(context, list):
            for subcontext in context:
                clean_context(subcontext)
        if isinstance(context, dict):
            to_pop = []
            for key, value in context.iteritems():
                if isinstance(value, np.ndarray):
                    to_pop.append(key)
                elif isinstance(value, dict):
                    clean_context(value)
            for key in to_pop:
                context.pop(key)


    percent_labels_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    # percent_links_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    percent_links_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    percent_unlabeled_range = [0.1, 0.2, 0.3, 0.4, 0.5]


    def get_lapsvm_grid(X, y, fit_kwargs):
        lapsvm_grid = {
            'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
            'gamma_A': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
            'gamma_I': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
        }
        return lapsvm_grid


    lapsvm_percents = list(itertools.izip_longest(percent_labels_range,
                                                  [],
                                                  percent_unlabeled_range,
                                                  fillvalue=0.0))

    links_and_labels = itertools.izip(percent_labels_range, percent_unlabeled_range)
    links_percents = [(labels, links, unlabeled) for (labels, unlabeled), links in
                      itertools.product(links_and_labels,
                                        percent_links_range[1:])]  # TODO: notice the hack


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


    links_estimator_tuple = estimator_tuple(
        name='Links',
        estimator=LinksClassifier(kernel='rbf', sampling='predefined', solver='tnc'),
        kwargs_func=lambda kw: kw,
        grid_func=links_grid_rbf)


    def links_context_task_generator(contexts):
        for context in contexts:
            X = context.pop('X_full')
            y = context.pop('y_full')
            train = context.pop('train')
            test = context.pop('test')
            clean_context(context)
            yield OrderedDict(context), {
                'estimator': links_estimator_tuple,
                'X': X,
                'y': y,
                'train': train,
                'test': test
            }


    if args.jobs == 1:
        mapper = itertools.imap
    else:
        mapper = mp.Pool(mp.cpu_count() if args.jobs == -1 else args.jobs).imap_unordered

    for ds_name, contexts in task_generator(links_grid_rbf, links_percents, do_random_search=False):
        for context in contexts:
            context['rs_iters'] = args.rs_iters
            context['rs_splits'] = args.rs_folds
            context['rs_test_size'] = args.rs_test_size
        print('%d contexts for %s' % (len(contexts), ds_name))
        runner = Runner(task=task,
                        task_generator=links_context_task_generator(contexts),
                        cacher=cacher,
                        mapper=mapper)
        runner.run()

    # generate lapsvm contexts and write them to disk
    # contexts_by_ds = task_generator(lapsvm_grid, lapsvm_percents)
    # print('generated %d contexts' % sum(map(len, contexts_by_ds.itervalues())))
    args.cv_folds = 15
    for ds_name, contexts in task_generator(get_lapsvm_grid, lapsvm_percents):

        mat_input_filename = 'data/lapsvm/lapsvm_%s.mat' % ds_name
        mat_output_filename = 'data/lapsvm/results/lapsvm_%s.mat' % ds_name
        if not os.path.isfile(mat_output_filename):
            if not os.path.isfile(mat_input_filename):
                print('saving %d contexts for MATLAB' % len(contexts))
                scipy.io.savemat(mat_input_filename, mdict={'stuff': contexts},
                                 do_compression=False)
            while not os.path.isfile(mat_output_filename):
                print('run MATLAB for %s' % ds_name)
                raw_input('done?')

        print('loading %d contexts from MATLAB' % len(contexts))
        clean_context(contexts)
        for c in contexts:
            # print(c.keys())
            c.pop('random_search')
        results = load_matlab_context(mat_output_filename, my_contexts=contexts)
        for context, result in itertools.izip(contexts, results):
            context['estimator'] = 'LapSVM'
            context['rs_iters'] = args.rs_iters
            context['rs_splits'] = args.rs_folds
            cacher.set(context, result)
        cacher.save()
    # contexts_by_ds = {}
    # # clean_context(contexts_by_ds)
    # gc.collect()
    #
    print('done')
