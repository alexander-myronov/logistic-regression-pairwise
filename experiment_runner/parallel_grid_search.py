"""
Module contains the implemetation of grid search with cross-validation,
parallelized using ipyparallel
"""
from collections import Sized, defaultdict
from functools import partial
import itertools

from sklearn.base import is_classifier, clone
from sklearn.cross_validation import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, indexable
import numpy as np

from grid import split_constraints_and_transforms, make_grid, apply_transforms
from process_grid import map_param

__author__ = 'amyronov'

from sklearn.grid_search import GridSearchCV, _CVScoreTuple, ParameterGrid


class GridSearchCVParallel(GridSearchCV):
    """
    GridSearchCV computes scores for every fold of the data for every point in hyperparameter grid.
    This GridSearchCV implementation distributes these tasks to an ipyparallel cluster
    It can be used the same way GridSeachCV from sklearn is used.
    """

    def __init__(self, *args, **kwargs):
        """
        GridSearchCVParallel accepts all the parameters for GridSearchCV from sklearn,
         but the has some additional parameters in the constructor
        callback: callable, optional
            Function to be called every every iteration
            Must accept 3 parameters: number of task done, total number of tasks and time elapsed (seconds)
        cacher: dictionary or CacherABC subclass, optional
        mapper: callable, optional: a map function (function, iterable). if not provided, itertools.imap will be used
        loader: callable, optional
            Parameterless function to be called on remote machines to load data.
            Must return X,y as a tuple, must handle its own imports
        fit_callback: callable, optional: a function to be called after _fit_and_score,
            parameters: index, {estimator, X, y, parameters, fit_params, train, test, scorer}
        """

        self.callback = kwargs.pop('callback', None)
        self.mapper = kwargs.pop('mapper', itertools.imap)
        self.cacher = kwargs.pop('cacher', {})
        self.loader = kwargs.pop('loader', None)
        self.fit_callback = kwargs.pop('fit_callback', None)
        self.max_iter = kwargs.pop('max_iter', 100)

        composite_grid = args[1]
        if isinstance(composite_grid, dict):
            grid = composite_grid
            self.transforms = {}
        else:
            grid, self.transforms = make_grid(*split_constraints_and_transforms(composite_grid),
                                              max_iter=self.max_iter)
        args = list(args)
        print('grid length: %d' % len(ParameterGrid(grid)))
        args[1] = grid
        args = tuple(args)

        super(GridSearchCVParallel, self).__init__(*args, **kwargs)

    def fit(self, X, y=None, x_is_index=False):
        """
        fit creates a task for every pair of folds and combination of hyperparameters in the grid
        it then distributes the tasks to ipyparallel view and waits for completion
        :param X: ndarray of data
        :param y: ndarray of target variables
        :param x_is_index: boolean variable to indicate that X is not the data itself,
            but the index of the data to be used on remote machines.
            Useful when sending the data by network is unfeasible
        """

        if not self.loader:
            self.loader = lambda: (X, y)

        parameter_iterable = ParameterGrid(self.param_grid)
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)

        if x_is_index and self.loader is None:
            raise ValueError('no loader given')

        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        train_test_parameters = ((train, test, apply_transforms(parameters, self.transforms)) \
                                 for parameters in parameter_iterable
                                 for train, test in cv)

        length = len(parameter_iterable) * len(cv)
        if self.callback:
            self.callback(len(self.cacher), length)

        if x_is_index:
            X_to_pass = X
            y_to_pass = y if self.loader is None else None
        else:
            if self.loader is not None:
                X_to_pass = None
                y_to_pass = None
            else:
                X_to_pass = X
                y_to_pass = y

        # print('sequences')

        # sequences = [
        #     train_test_parameters,
        #     [clone(base_estimator)] * length,
        #     [X_to_pass] * length,
        #     [y_to_pass] * length,
        #     [self.verbose] * length,
        #     [self.fit_params] * length,
        #     [True] * length,
        #     [self.scorer_] * length,
        #     [x_is_index] * length,
        # ]

        f = partial(GridSearchCVParallel.my_fit_and_score,
                    estimator=base_estimator,
                    X=X_to_pass,
                    y=y_to_pass,
                    fit_params=self.fit_params,
                    scorer=self.scorer_,
                    x_is_index=x_is_index,
                    loader=self.loader,
                    fit_callback=self.fit_callback)

        iterable = itertools.ifilter(lambda (i, ttp): i not in self.cacher,
                                     enumerate(train_test_parameters))



        results_by_params = defaultdict(list)
        for id, result in self.cacher.iteritems():
            score, test_size, time, params = result
            results_by_params[frozenset(map(map_param, params.iteritems()))].append(score)

        for index, result in self.mapper(f, iterable):
            self.cacher[index] = result
            score, test_size, time, params = result
            results_by_params[frozenset(map(map_param, params.iteritems()))].append(score)
            if self.callback:
                best_scores = next(iter(sorted(itertools.ifilter(lambda scores:
                                                                 len(scores) == len(cv), results_by_params.values()),
                                               key=lambda scores: np.mean(scores),
                                               reverse=True)), [0])

                self.callback(1, length, description='%.3f+-%.3f' % (np.mean(best_scores), np.std(best_scores)))

        # assert len(self.cacher) == length and (np.array(self.cacher.keys()) == np.arange(length)).all()

        # out = self.cacher.values()
        #
        # # Out is a list of triplet: score, estimator, n_test_samples
        # n_fits = len(out)
        # n_folds = len(cv)
        #
        # scores = list()
        # grid_scores = list()
        # for grid_start in range(0, n_fits, n_folds):
        #     n_test_samples = 0
        #     score = 0
        #     all_scores = []
        #     for this_score, this_n_test_samples, _, parameters in \
        #             out[grid_start:grid_start + n_folds]:
        #         all_scores.append(this_score)
        #         if self.iid:
        #             this_score *= this_n_test_samples
        #             n_test_samples += this_n_test_samples
        #         score += this_score
        #     if self.iid:
        #         score /= float(n_test_samples)
        #     else:
        #         score /= float(n_folds)
        #     scores.append((score, parameters))
        #     # TODO: shall we also store the test_fold_sizes?
        #     grid_scores.append(_CVScoreTuple(
        #         parameters,
        #         score,
        #         np.array(all_scores)))

        grid_scores = []
        for set_params, all_scores in results_by_params.iteritems():
            grid_scores.append(_CVScoreTuple(
                dict(set_params),
                np.mean(all_scores),
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

    @staticmethod
    def my_fit_and_score(index_train_test_parameters,
                         estimator=None,
                         X=None,
                         y=None,
                         fit_params=None,
                         scorer=None,
                         x_is_index=True,
                         loader=None,
                         fit_callback=None):
        """
        function which represents a single task execution for GridSeearchCVParallel,
        is executed on remote machines
        :param index_train_test_parameters: tuple of 4 (index of the task, train index, test index, hyperparameters)
        :param estimator: sklearn's BaseEstimator subtype
        :param X: the data, or index (if x_is_index is True), or None
        :param y: the target variable
        :param fit_params: parameters for fit function of estimator
        :param scorer: sklearn scorer(estimator, X, y)
        :param x_is_index: True if x is to be used to get a subset of the data
        :param loader: function to load data on remote machines
        :return: tuple of 2 (index, result of sklearn.cross_validation._fit_and_score)
        """
        from sklearn.cross_validation import _fit_and_score
        gs_index, (train, test, parameters) = index_train_test_parameters

        if x_is_index:
            index = X
            X = None
        if X is None:
            if loader is None:
                raise ValueError('loader is missing, X is None')
            X, y = loader()

        if x_is_index:
            X = X[index]
            y = y[index]

        # setup_kfold_patch(10)

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


if __name__ == '__main__':
    pass