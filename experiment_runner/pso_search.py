import random
from functools import partial

import numpy as np
import optunity
import pandas as pd
import itertools
import warnings

from sklearn import clone
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import _CVScoreTuple
from grid import make_structured_space, split_constraints_and_transforms, apply_transforms


class PSOSearch(object):
    def __init__(self,
                 estimator,
                 grid,
                 cv,
                 scoring,
                 callback,
                 cacher,
                 loader,
                 mapper,
                 fit_callback,
                 max_iter=100):
        self.estimator = estimator
        self.space, self.transforms = make_structured_space(*split_constraints_and_transforms(grid))
        self.cv = cv
        self.scoring = scoring
        self.cacher = cacher
        self.loader = loader
        self.mapper = mapper
        self.fit_callback = fit_callback
        self.callback = callback
        self._best_score = None
        self._grid_scores = None
        self.max_iter = max_iter
        self._best_params = None

    def fit(self, X, y):
        if not self.loader:
            self.loader = lambda: (X, y)

        est = self.estimator
        scorer = self.scoring
        cv = self.cv
        loader = self.loader

        callback = self.callback
        length = len(cv) * self.max_iter
        if callback:
            callback(0, length)
        transforms = self.transforms
        cacher = self.cacher
        fit_callback = self.fit_callback
        mapper = self.mapper

        cv_scores = {}

        def fit_func(**params):
            params = apply_transforms(params, transforms)
            base_id = len(cv_scores) * len(cv)

            scores = PSOSearch.cross_val_score(
                base_index=base_id,
                estimator=est,
                parameters=params,
                loader=loader,
                cv=cv,
                scorer=scorer,
                fit_callback=fit_callback,
                cacher=cacher,
                callback=callback,
                mapper=mapper
            )

            cv_score = _CVScoreTuple(
                params,
                np.mean(scores),
                scores)
            cv_scores[base_id] = cv_score
            best_score_params = cv_scores.values()[
                np.argmax(np.array(map(lambda score: score.mean_validation_score, cv_scores.itervalues())))]
            best_score_mean = best_score_params.mean_validation_score
            best_score_std = np.std(best_score_params.cv_validation_scores)
            if callback:
                callback(description='%.3f+-%.3f' % (best_score_mean, best_score_std))
            return scores.mean()

        np.random.seed(1)
        random.seed(1)
        res, optimize_results, solver_info = optunity.maximize_structured(fit_func, self.space, num_evals=self.max_iter)

        self._best_score = optimize_results[0]
        self._grid_scores = cv_scores
        self._best_params = res

    @staticmethod
    def get_from_cacher_safe(cacher, id, desired_params):
        if id not in cacher:
            return None
        _, _, _, stored_params = cacher[id]
        if set(desired_params.iteritems()) == set(stored_params.iteritems()):
            return cacher[id]

    @staticmethod
    def cross_val_score(base_index,
                        estimator,
                        parameters,
                        loader,
                        cv,
                        scorer,
                        fit_callback,
                        cacher,
                        callback,
                        mapper=itertools.imap):

        scores = {}
        list_to_compute = []
        for i, (train, test) in enumerate(cv):
            index = i + base_index
            result_cache = PSOSearch.get_from_cacher_safe(cacher, index, parameters)
            if result_cache:
                scores[index] = result_cache[0]
                if callback:
                    callback(1, 0)
            else:
                list_to_compute.append((index, (train, test, parameters)))

        f = partial(PSOSearch.my_fit_and_score,
                    estimator=estimator,
                    fit_params=None,
                    scorer=scorer,
                    loader=loader,
                    fit_callback=fit_callback)

        for index, result in mapper(f, list_to_compute):
            scores[index] = result[0]
            cacher[index] = result
            if callback:
                callback(1, 0)

        return np.array(scores.values())

    @staticmethod
    def my_fit_and_score(index_train_test_parameters,
                         estimator=None,
                         loader=None,
                         fit_params=None,
                         scorer=None,
                         fit_callback=None):
        from sklearn.cross_validation import _fit_and_score
        gs_index, (train, test, parameters) = index_train_test_parameters

        if loader is None:
            raise ValueError('loader is missing')
        X, y = loader()
        estimator = clone(estimator)
        result = _fit_and_score(estimator=estimator,
                                X=X,
                                y=y,
                                verbose=False,
                                parameters=parameters,
                                fit_params=fit_params,
                                return_parameters=True,
                                train=train,
                                test=test,
                                scorer=scorer)

        if fit_callback:
            fit_callback(gs_index,
                         {
                             'estimator': estimator,
                             'X': X,
                             'y': y,
                             'parameters': parameters,
                             'fit_params': fit_params,
                             'train': train,
                             'test': test,
                             'scorer': scorer
                         })

        return gs_index, result

    @property
    def best_score_(self):
        if self._best_score is None:
            raise Exception
        return self._best_score

    @property
    def grid_scores_(self):
        if self._grid_scores is None:
            raise Exception
        return self._grid_scores

    @property
    def best_params_(self):
        if self._best_params is None:
            raise Exception
        return self._best_params
