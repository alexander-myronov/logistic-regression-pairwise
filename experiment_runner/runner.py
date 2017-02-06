import itertools
import traceback

import numpy as np
import os
import pandas as pd
import sys
from dill import dill
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline, FeatureUnion

from experiment_runner.pso_search import PSOSearch
from experiment_runner.tqdm_callback import TqdmCallback
from experiment_runner.caching import MultipleFilesCacher, RemoteMultipleFilesCacher
from experiment_runner.parallel_grid_search import GridSearchCVParallel

from summary import get_summary


def estimators_equal(est1, est2):
    if type(est1) != type(est2):
        return False

    params1 = est1.get_params(deep=True)
    params2 = est2.get_params(deep=True)

    for key1 in params1.keys():
        if key1 not in params2:
            return False
        val1 = params1[key1]
        val2 = params1[key1]
        if isinstance(val1, BaseEstimator):
            if not estimators_equal(val1, val2):
                return False
        else:
            if val1 != val2:
                return False
    return True


def grids_equal(grid1, grid2):
    if isinstance(grid1, list):
        if not isinstance(grid2, list):
            return False
        for subgrid1, subgrid2 in itertools.izip(grid1, grid2):
            if not grids_equal(subgrid1, subgrid2):
                return False
    for (name1, value1), (name2, value2) in itertools.izip(
            sorted(grid1.iteritems(), key=lambda (k, v): k),
            sorted(grid2.iteritems(), key=lambda (k, v): k)):
        if name1 != name2:
            return False
        if isinstance(value1, tuple):
            if not isinstance(value2, tuple) or len(value1) != len(value2):
                return False
            if value1[0] != value2[0] or value1[1] != value2[1]:
                return False
        else:
            for v1, v2 in itertools.izip(set(value1), set(value2)):
                if isinstance(v1, object) and isinstance(v2, object):
                    continue
                if v1 != v2:
                    return False
    return True


def cv_equal(cv1, cv2):
    if len(cv1) != len(cv2):
        return False

    for (tr1, tst1), (tr2, tst2) in itertools.izip(cv1, cv2):
        if not (tr1 == tr2).all() or not (tst1 == tst2).all():
            return False
    return True


def validate_cache(meta1, meta2):
    if not estimators_equal(meta1['estimator'], meta2['estimator']):
        raise Exception('Cached estimator not equal')
    if not grids_equal(meta1['grid'], meta2['grid']):
        raise Exception('Cached grid not equal')
    if not cv_equal(meta1['cv'], meta2['cv']):
        raise Exception('Cached cv not equal')
    if meta1['X_shape'] != meta2['X_shape']:
        raise Exception('Cached X shape not equal')
    if (meta1['y_unique'] != meta2['y_unique']).all():
        raise Exception("Cached y not equal")
    if (meta1['search'] != meta2['search'] or \
                    set(meta1['search_kwargs'].iteritems()) != set(
                    meta2['search_kwargs'].iteritems())):
        raise Exception("Cached search not equal: cached=%s, used=%s" %
                        (str(meta1['search']) + ' ' + str(meta1['search_kwargs']),
                         str(meta2['search']) + ' ' + str(meta2['search_kwargs'])))


class ExperimentRunner(object):
    def __init__(self,
                 experiment_name,
                 estimators,
                 dataset,
                 cv,
                 scorer,
                 dir,
                 mapper=itertools.imap,
                 search_algorithm=GridSearchCVParallel,
                 search_algorithm_kwargs={'iid': False, 'refit': False}):
        self.experiment_name = experiment_name
        self.mapper = mapper
        self.dataset = dataset
        self.estimators = estimators
        self.scorer = scorer
        self.cv = cv

        self.dir = self.prepare_dir(dir)
        self.search = search_algorithm
        self.search_kwargs = search_algorithm_kwargs
        self.results = {}

    def prepare_dir(self, dir):
        if not dir.endswith('/'):
            dir += '/'
        if not os.path.isdir(dir):
            os.makedirs(dir)
        else:
            pass
            # estimators = None
            # try:
            #     with open(dir + 'estimators.pkl', 'rb') as f:
            #         estimators = dill.load(f)
            # except:
            #     pass
            # if estimators:
            #     for (name1, est1, grid1), (name2, est2, grid2) in itertools.izip_longest(estimators):
            #         # raise NotImplementedError("estimators equality comparison")
            #         if type(est1) !=  type(est2):
            #             raise Exception("analysis in %s was initiated using different ")

        return dir

    def run(self):

        if callable(self.dataset):
            X, y = self.dataset()
            loader = self.dataset
        else:
            X, y = self.dataset
            loader = lambda: (X, y)

        for name, estimator, grid in self.estimators:
            print(name)
            cache_dir = '%s/%s/' % (self.dir, name)

            if hasattr(self.cv, '__len__'):
                cv = list(self.cv)
            elif callable(self.cv):
                cv = list(self.cv(y))
            else:
                raise NotImplementedError()

            meta = {
                'X_shape': X.shape,
                'y_unique': np.unique(y),
                'cv': cv,
                'name': name,
                'estimator': estimator,
                'grid': grid,
                'search': self.search,
                'search_kwargs': self.search_kwargs
            }

            old_meta = None
            meta_filename = cache_dir + 'meta.pkl'
            if os.path.exists(meta_filename):
                try:
                    with open(meta_filename, 'rb') as f:
                        old_meta = dill.load(f)
                except:
                    pass

            if old_meta:
                validate_cache(meta, old_meta)

            cacher = MultipleFilesCacher(cache_dir, flush_every_n=5)

            callback = TqdmCallback()

            def record_metadata(index, fit_arguments):
                meta_cacher = RemoteMultipleFilesCacher(cache_dir, flush_every_n=1,
                                                        file_name_source=lambda
                                                            key: '%d_meta.pkl' % key)
                X = fit_arguments.pop('X')
                y = fit_arguments.pop('y')
                estimator = fit_arguments.pop('estimator')
                test = fit_arguments['test']
                y_pred = estimator.predict(X[test])
                fit_arguments['y_pred'] = y_pred
                fit_arguments['y_true'] = y[test]
                meta_cacher[index] = fit_arguments

            search = self.search(estimator,
                                 grid,
                                 scoring=self.scorer,
                                 cv=cv,
                                 callback=callback,
                                 cacher=cacher,
                                 loader=loader,
                                 mapper=self.mapper,
                                 fit_callback=record_metadata,
                                 **self.search_kwargs)

            try:

                with open(meta_filename, 'wb') as f:
                    dill.dump(meta, f, -1)

                search.fit(X, y)

                cacher.save()

                print(name, search.best_score_)

                summary = get_summary(self.experiment_name, X, y, cv, estimator,
                                      search.grid_scores_)
                print(summary)

                meta['best_score'] = search.best_score_
                meta['grid_scores'] = search.grid_scores_
                meta['summary'] = summary
                self.results[name] = meta

                with open(meta_filename, 'wb') as f:
                    dill.dump(meta, f, -1)

            except Exception as e:

                e_type, e_value, e_tb = sys.exc_info()
                tb = ''.join(traceback.format_tb(e_tb))
                print(e_type, e_value)
                print(tb)

            summary = '\n'.join(map(lambda m: m['summary'], self.results.itervalues()))
            with open(self.dir + 'summary.txt', 'w') as f:
                f.write(summary)
