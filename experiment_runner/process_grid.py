import argparse
import cPickle
import operator
import os
import itertools
import traceback
from collections import defaultdict, OrderedDict
import re

import imp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dill import dill
from sklearn.grid_search import ParameterGrid, _CVScoreTuple

from experiment_runner.caching import SingleFileCacher, MultipleFilesCacher
from grid import to_int


def plot_2d_slice(score_series, values1_series, values2_series, title, ax=None, print_score=True):
    """
    plot 2d slice of grid search results
    :param score_series: mean validation score
    :param values1_series: pd.Series of 1st dimension values
    :param values2_series: pd.Series of 2nd dimension values
    :param title: plot title
    :param ax: matplotlib.Axes to plot onto
    :param print_score: True to print score (already represented by colour on the plot)
    :return: figure
    """
    values1_series = values1_series.fillna(value='None')
    values2_series = values2_series.fillna(value='None')
    index = pd.MultiIndex.from_arrays([values1_series, values2_series])
    score_series = pd.Series(score_series.values, index=index)
    scores = score_series.unstack()

    scores.fillna(0, inplace=True)

    if ax is None:
        fig, ax = plt.subplots()
        # ax = plt

    # scores = score_series.values
    # ax.scatter(values1_series.values, values2_series.values, c=score_series.values, cmap=plt.cm.hot, s=100, marker='s')
    # r = None
    r = ax.imshow(np.round(scores.values, 3), interpolation='nearest', cmap=plt.cm.hot, vmin=0,
                  vmax=scores.values.max())

    ax.set_xticks(np.arange(len(scores.columns)))
    ax.set_xticklabels(scores.columns)
    ax.set_yticks(np.arange(len(scores.index)))
    ax.set_yticklabels(scores.index)
    # ax.set_title(title)
    ax.set_xlabel(re.sub('\w+__', '', values2_series.name))
    ax.set_ylabel(re.sub('\w+__', '', values1_series.name))

    if print_score:
        for (j, i), v in np.ndenumerate(scores):
            c = (0, 0, 0)
            if v <= 0.5:
                c = (0.8, 0.8, 0.8)
            ax.text(i, j, '%.3g' % np.round(v, 3), va='center', ha='center', color=c)
    return r


def ncr(n, r):
    r = min(r, n - r)
    if r == 0: return 1
    numer = reduce(operator.mul, xrange(n, n - r, -1))
    denom = reduce(operator.mul, xrange(1, r + 1))
    return numer // denom


def plot_max_slice(params_scores):
    """
    slice grid search results hypercube by all possible pairs of 2 dimensions
    :param params_scores: grid search results
    :return: list of figures
    """
    # add_missing_params(params_list)

    params = map(lambda p: p.parameters, params_scores)

    df = pd.DataFrame(params)
    param_columns = filter(lambda c: len(df[c].unique()) > 1, list(df.columns))
    single_value_columns = filter(lambda c: len(df[c].unique()) == 1, list(df.columns))
    title = ',\n'.join(map(lambda c: '%s=%s' % (re.sub('\w+__', '', c), df[c].unique()[0]), single_value_columns))

    scores = map(lambda p: p.cv_validation_scores.mean(), params_scores)
    scores = np.array(scores)

    stds = map(lambda p: p.cv_validation_scores.std(), params_scores)
    stds = np.array(stds)
    # scores[scores > 0.6] = 0.5
    df.loc[:, 'score'] = pd.Series(scores)
    df.fillna(value='None', inplace=True)

    best_index = df['score'].argmax()
    print(title.replace('\n', ' '),
          '%.2g+-%.2g' % (np.round(df['score'].values[best_index], 2), np.round(stds[best_index], 2)))

    if len(param_columns) == 0:
        fig, ax = plt.subplots()
        ax.hist(params_scores[0].cv_validation_scores)
        raise Exception('no variable params')
    if len(param_columns) == 1:
        fig, ax = plt.subplots()
        r = plot_2d_slice(df['score'],
                          df[param_columns[0]],
                          pd.Series(data=[' '] * len(df['score']), name=' '), title, ax=ax)
        plt.colorbar(r)

        #
        # ax.plot(df[param_columns[0]], df['score'])
        #
        # ax.set_xticks(np.arange(len(df[param_columns[0]])))
        # ax.set_xticklabels(df[param_columns[0]])
        # ax.set_yticks(np.arange(len(df['score'])))
        # ax.set_yticklabels(df['score'])
        # ax.set_title(title)
        # ax.set_xlabel(param_columns[0])
        # ax.set_ylabel('score')
        # figures.append(fig)
    elif len(param_columns) == 2:
        fig, ax = plt.subplots()
        r = plot_2d_slice(df['score'], df[param_columns[0]], df[param_columns[1]], title, ax=ax)
        plt.colorbar(r)

    else:
        l = ncr(len(param_columns), 2)
        if l < 5:
            fig, ax = plt.subplots(1, l)
        else:
            fig, ax = plt.subplots(2, int(np.ceil(l / 2.0)))
            ax = ax.ravel()
        for i, (col1, col2) in enumerate(itertools.combinations(sorted(param_columns), 2)):
            ax1 = ax[i]
            remaining_columns = sorted(set(param_columns).difference([col1, col2]))

            cur_title = title
            slice = df
            for col in remaining_columns:
                slice_value = df.ix[best_index, col]
                slice = slice[slice[col] == slice_value]
                cur_title += ',\n%s=%s' % (re.sub('\w+__', '', col), slice_value)

            values1 = slice[col1]
            values2 = slice[col2]
            r = plot_2d_slice(slice['score'], values1, values2, cur_title, ax=ax1)
        plt.colorbar(r)

        # fig.savefig('data/', bbox_inches='tight')
        # fig.savefig()
    return fig


def remove_redundant_param(results, param_name):
    if len(filter(lambda p: param_name in p.parameters, results)):
        any_value = map(lambda p: p.parameters[param_name], results)[0]
        results = filter(lambda score: score.parameters[param_name] == any_value,
                         results)
        for rr in results:
            del rr.parameters[param_name]
    return results


def split_svm_results_by_kernel(grid_results):
    """
    split svm grid search results by kernel type (linear, poly2, poly3, rbf)
    :param grid_results: grid search results
    :return: dictionary: kernel name -> kernel results
    """
    names = set(reduce(operator.add, map(lambda p: p.parameters.keys(), grid_results)))

    kernel_column = next(itertools.ifilter(lambda n: n.endswith('kernel'), names), '')
    degree_column = next(itertools.ifilter(lambda n: n.endswith('degree'), names), '')
    gamma_column = next(itertools.ifilter(lambda n: n.endswith('gamma'), names), '')

    degree_range = set(sorted(map(lambda r: r.parameters[degree_column],
                                  filter(lambda r: degree_column in r.parameters, grid_results))))

    splits = OrderedDict()

    linear_results = filter(lambda score: score.parameters[kernel_column] == 'linear',
                            grid_results)

    linear_results = remove_redundant_param(linear_results, degree_column)
    linear_results = remove_redundant_param(linear_results, gamma_column)

    if len(linear_results) > 0:
        splits['linear'] = linear_results

    for degree in degree_range:
        poly = filter(lambda score: score.parameters[kernel_column] == 'poly' and \
                                    score.parameters[degree_column] == degree,
                      grid_results)
        if len(poly) > 0:
            splits['poly%d' % degree] = poly

    # poly_results = filter(lambda score: score.parameters[kernel_column] == 'poly',
    #                    grid_results)
    # if len(poly_results) > 0:
    #     splits.append(poly_results)

    rbf_results = filter(lambda score: kernel_column in score.parameters and score.parameters[kernel_column] == 'rbf',
                         grid_results)

    rbf_results = remove_redundant_param(rbf_results, degree_column)

    if len(rbf_results) > 0:
        splits['rbf'] = rbf_results

    return splits


def iterate_subfolders_meta(dir):
    for dirpath, dirnames, filenames in os.walk(dir):
        meta_filename = dirpath + '/meta.pkl'
        try:
            with open(meta_filename, 'rb') as meta_f:
                meta = dill.load(meta_f)
                yield dirpath, meta['name'], meta['estimator'], meta['grid']
        except Exception as e:
            pass


def map_param(k_v):
    k, v = k_v
    if isinstance(v, (int, long, float, str)):
        return k, v
    return k, str(type(v))


def cached_results_to_cvscores(cache_results):
    results_by_params = defaultdict(list)
    for score, test_set_size, time, params in cache_results:
        results_by_params[frozenset(map(map_param, params.iteritems()))].append(score)

    grid_results = []
    for set_params, all_scores in results_by_params.iteritems():
        grid_results.append(_CVScoreTuple(
            dict(set_params),
            np.mean(all_scores),
            np.array(all_scores)))
    return grid_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= \
                                         'Script to present the results obtained by grid_evaluation.py')

    parser.add_argument('--estimators_file', type=str,
                        help='python file with list of 3tup (name, estimator, grid) called `estimators`')

    parser.add_argument('--data_dir', type=str,
                        default=r'data/grid_search',
                        help='local results directory')

    parser.add_argument('--width', type=float, default=20,
                        help='picture width (in)')
    parser.add_argument('--height', type=float, default=15,
                        help='picture height (in)')

    parser.add_argument('--dpi', type=int, default=100,
                        help='picture dpi')

    args = parser.parse_args()
    print(args)

    # assert args.estimators_file is not None
    # estimators_module = imp.load_source('estimators', args.estimators_file)
    # estimators = estimators_module.estimators

    base_dir = args.data_dir
    figures = {}

    for dir, name, estimator, grid in iterate_subfolders_meta(base_dir):
        try:
            print(dir, name)

            filename = '%s/%s.pkl' % (dir, name)

            if os.path.isfile(filename):
                grid_results = cPickle.load(open(filename, 'rb'))
            else:
                cache_filename = '%s_cache.pkl' % (dir)
                if os.path.isfile(cache_filename):
                    cache_results = SingleFileCacher(cache_filename).values()
                else:
                    cache_folder = dir
                    if os.path.isdir(cache_folder):
                        cache_results = MultipleFilesCacher(cache_dir=cache_folder).values()
                    else:
                        print('no results found for %s' % name)
                        continue

                grid = ParameterGrid(grid)

                grid_scores= cached_results_to_cvscores(cache_results)

            if not os.path.exists('%s/images/' % base_dir):
                os.makedirs('%s/images/' % base_dir)

            has_kernel_param = next(itertools.ifilter(lambda n: n.endswith('kernel'),
                                                      set(itertools.chain(*itertools.imap(lambda r:
                                                                                          r.parameters.keys(),
                                                                                          grid_results)))),
                                    False)
            if 'SVM' in name or has_kernel_param:
                for kernel, res in split_svm_results_by_kernel(grid_results).iteritems():
                    fig = plot_max_slice(res)
                    figures['%s_%s' % (name, kernel)] = fig

            else:
                fig = plot_max_slice(grid_results)
                figures['%s' % (name)] = fig

        except Exception as e:
            pass
            # traceback.print_exc()

    print('saving pictures')
    for name, fig in figures.iteritems():
        fig.set_size_inches(args.width, args.height)
        fig.savefig('%s/images/%s.png' % (base_dir, name), bbox_inches='tight', dpi=300)
    plt.show()
