import itertools
import random
import types

import numpy as np
from sklearn import clone
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import ParameterGrid
import math
import operator

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, make_scorer, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def get_svm_grid():
    grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': [20, 10, 5, 1, 0.01, 0.001, 0.0001, 1e-5, 1e-7],
        'degree': [2, 3]
    }

    grid_sequence = []

    grid_sequence.append({'kernel': ['linear'], 'C': grid['C']})
    grid_sequence.append({'kernel': ['rbf'], 'C': grid['C'], 'gamma': grid['gamma']})
    grid_sequence.append({'kernel': ['poly'], 'C': grid['C'], 'gamma': grid['gamma'], 'degree': grid['degree']})

    return grid_sequence


def get_svm_grid_short():
    grid = [{
        'C': [0.01, 0.1],
        # 'degree': [2, 3, 4, 5]
    }]
    return grid


def get_svm_grid_with_prefix(prefix):
    grid = get_svm_grid()
    newgrid = []
    for elem in grid:
        newgrid.append({prefix + k: v for k, v in elem.iteritems()})
    return newgrid


def append_grid(grid_sequence, grid_dict):
    """
    >>> import pprint
    >>> pprint.pprint(append_grid([{'a':1, 'b':2}, {'a':10, 'b':20}], {'c': 5}))
    [{'a': 1, 'b': 2, 'c': 5}, {'a': 10, 'b': 20, 'c': 5}]

    """
    for seq in grid_sequence:
        seq.update(grid_dict)
    return grid_sequence


id_transform = lambda name, value: (name, value)


def to_int(name, value):
    return name, int(round(value))


type_transforms = {
    int: lambda str, v: int(str),
    float: lambda str, v: float(str),
    types.NoneType: lambda str, v: None,
    str: lambda str, v: str,
}


def make_structured_node(name, choices, terminal_value):
    name_transforms = {}
    node = {}
    for choice in choices:
        if not isinstance(choice, str):
            name_transforms[name, str(choice)] = choice
        node[str(choice)] = terminal_value
    return {name: node}, name_transforms


def make_structured_space(grid, constraints, transforms):
    name_transforms = {}
    terminal_value = constraints if len(constraints) > 0 else None
    for name, choices in grid.iteritems():
        node, more_transforms = make_structured_node(name, choices, terminal_value)
        terminal_value = node
        name_transforms.update(more_transforms)

        # transforms[name] = lambda name, value: more_transforms[(name, value)]

    for name in itertools.imap(operator.itemgetter(0), name_transforms.iterkeys()):
        transforms[name] = lambda name, value: \
            (name, name_transforms[(name, value)]) if (name, value) in name_transforms else (name, value)

    return terminal_value, transforms


def make_grid(grid, constraints, transforms, max_iter=100):
    """
    >>> import pprint
    >>> pprint.pprint(list(ParameterGrid(make_grid({'a':[1], 'b':[3,4]}, {'c':(0, 10)}, max_iter=4))))
    [{'a': 1, 'b': 3, 'c': 0.0},
     {'a': 1, 'b': 3, 'c': 10.0},
     {'a': 1, 'b': 4, 'c': 0.0},
     {'a': 1, 'b': 4, 'c': 10.0}]
    """
    final_grid = dict(grid)
    iterations_required = reduce(operator.mul, itertools.imap(len, final_grid.itervalues()), 1)
    if iterations_required > max_iter:
        raise Exception("Too few iterations to cover grid")
    if len(constraints) == 0:
        max_iter_by_dimension = []
    else:
        if len(constraints) == 1:
            max_iter_by_dimension = max_iter / iterations_required
            max_iter_by_dimension = [max_iter_by_dimension] * len(constraints)
        else:
            max_iter_by_dimension = max(int(math.log(max_iter / iterations_required, len(constraints))), 1)
            max_iter_by_dimension = [max_iter_by_dimension] * len(constraints)
        max_iter_reached = [False] * len(constraints)

        for i in xrange(len(constraints)):
            name, (value_min, value_max) = constraints.items()[i]
            while max_iter_by_dimension[i] > 0:
                values = np.linspace(value_min, value_max, max_iter_by_dimension[i])
                values_transformed = np.array(map(lambda value:
                                                  transforms[name](name, value)[1] if name in transforms else value,
                                                  values))
                if len(values_transformed) > len(np.unique(values_transformed)):
                    max_iter_by_dimension[i] -= 1
                    max_iter_reached[i] = True
                else:
                    break
            if max_iter_by_dimension[i] == 0:
                raise Exception("Any value for %s is not possible" % name)

        for i in itertools.chain.from_iterable(itertools.repeat(xrange(len(constraints)))):
            if all(max_iter_reached):
                break
            if max_iter_reached[i]:
                continue
            max_iter_by_dimension[i] += 1
            total_constraints_iterations = reduce(operator.mul, max_iter_by_dimension, 1) * iterations_required

            name, (value_min, value_max) = constraints.items()[i]
            values = np.linspace(value_min, value_max, max_iter_by_dimension[i])
            values_transformed = np.array(map(lambda value:
                                              transforms[name](name, value)[1] if name in transforms else value,
                                              values))
            if total_constraints_iterations > max_iter or \
                            len(values_transformed) > len(np.unique(values_transformed)):
                max_iter_by_dimension[i] -= 1
                max_iter_reached[i] = True

    for (name, (value_min, value_max)), n_iter in itertools.izip(constraints.iteritems(), max_iter_by_dimension):
        final_grid[name] = list(np.linspace(value_min, value_max, n_iter))
    return final_grid, transforms


def split_constraints_and_transforms(combined_grid):
    grid = {}
    constraints = {}
    transforms = {}

    if isinstance(combined_grid, list):
        raise NotImplementedError("TODO: splitting combined grids")

    for name, value_range in combined_grid.iteritems():
        if isinstance(value_range, tuple):
            if callable(value_range[-1]):
                transforms[name] = value_range[-1]
                if len(value_range) == 2:
                    grid[name] = value_range[0]
                elif len(value_range) == 3:
                    constraints[name] = (value_range[0], value_range[1])
                else:
                    raise NotImplementedError
            elif len(value_range) == 2:
                constraints[name] = (value_range[0], value_range[1])
        else:
            grid[name] = value_range
    return grid, constraints, transforms


def apply_transforms(params, transforms):
    # return dict([transforms[name](name, value) if name in transforms else (name, value)
    #              for name, value in params.iteritems()])

    result = {}
    for name, value in params.iteritems():
        if name in transforms:
            name, value = transforms[name](name, value)
        result[name] = value
    return result


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    # from estimators_regression import estimators
    #
    # name, estimator, grid = estimators[0]
    # X, y = loadXy('data/SLEDAI.hdf')  # load_svmlight_file('svm_test/liver-disorders_scale.libsvm')
    # X = X.toarray()
    # cv = StratifiedKFold(y, n_folds=5)
    # scoring = make_scorer(accuracy_score)
    #
    # from experiment_runner.runner import ExperimentRunner
    #
    # runner = ExperimentRunner(
    #     'PSO test',
    #     estimators=[(name, estimator, grid)],
    #     cv=cv,
    #     scorer=scoring,
    #     dataset=(X, y),
    #     dir='data/PSO_test/'
    # )
    #
    # runner.run()
    #
    # print(runner.results)

    combined_grid = {
        'regression__n_estimators': (50, 550, to_int),
        'regression__max_depth': (5, 10, to_int),
    }
    grid, transforms = make_grid(*split_constraints_and_transforms(combined_grid))
    #
    #
    # def func(**params):
    #
    #     params = apply_transforms(params, transforms)
    #     print('params=%s' % params)
    #     est = clone(estimator)
    #     est.set_params(**params)
    #
    #     scores = cross_val_score(est, X, y, scoring=scoring, cv=cv)
    #
    #     return scores.mean()
    #
    # np.random.seed(1)
    # random.seed(1)
    # res, optimize_results, solver_info= optunity.maximize_structured(func, space, num_evals=3)
    # print('best params=%s' % res)
    # print('best result=%f' % optimize_results[0])
    # print('solver info=%s' % solver_info)
