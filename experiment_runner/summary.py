from StringIO import StringIO
from sys import stdout

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from ttest import f_ttest

from feauture_filtering import SecondaryOnlyActiveFeatures

import numpy as np

from kfold_repeat import KFoldRepeatStable


def get_parameter_range(grid, name):
    param_range = set()
    for point in grid:
        if name in point:
            param_range.add(point[name])
    return sorted(list(param_range))


def get_estimator_description(estimator, grid, outfile=stdout, print_prefix='', grid_prefix=''):
    if isinstance(estimator, Pipeline):
        outfile.write(print_prefix + 'pipeline: \n')
        for name, step in estimator.steps:
            get_estimator_description(step, grid, outfile, print_prefix=print_prefix + '\t',
                                      grid_prefix=grid_prefix + name + '__')
        return
    if isinstance(estimator, FeatureUnion):
        outfile.write(print_prefix + 'union: \n')
        for name, step in estimator.transformer_list:
            get_estimator_description(step, grid, outfile, print_prefix=print_prefix + '\t',
                                      grid_prefix=grid_prefix + name + '__')
        return
    if isinstance(estimator, SelectKBest):
        test_f_dic = {
            f_classif: 'F-test',
            f_ttest: 't-test',
        }

        k_range = str(get_parameter_range(grid, grid_prefix + 'k'))

        outfile.write(
            print_prefix + 'feature selection: %s features by %s\n' % (k_range, test_f_dic[estimator.score_func]))
        return
    if isinstance(estimator, SecondaryOnlyActiveFeatures):
        outfile.write(print_prefix + \
                      'feature selection: secondary only controls - %s sigmas from mean in at least %s samples\n' %
                      (str(get_parameter_range(grid, grid_prefix + 'sigma_distance_threshold')),
                       str(get_parameter_range(grid, grid_prefix + 'sample_percent_threshold'))))
        return
    if isinstance(estimator, SVC):
        outfile.write(print_prefix + \
                      'classification: svm - C: %s, gamma: %s, kernel: %s\n' %
                      (str(get_parameter_range(grid, grid_prefix + 'C')),
                       str(get_parameter_range(grid, grid_prefix + 'gamma')),
                       str(get_parameter_range(grid, grid_prefix + 'kernel'))))
        return
    if isinstance(estimator, RandomForestClassifier):
        outfile.write(print_prefix + \
                      'classification: Random forest - trees: %s, max_depth: %s, min_samples_leaf: %s\n' %
                      (str(get_parameter_range(grid, grid_prefix + 'n_estimators')),
                       str(get_parameter_range(grid, grid_prefix + 'max_depth')),
                       str(get_parameter_range(grid, grid_prefix + 'min_samples_leaf'))))
    if isinstance(estimator, KernelPCA):
        outfile.write(print_prefix + \
                      'decomposition: pca - n_components: %s, kernel: %s, degree: %s\n' %
                      (str(get_parameter_range(grid, grid_prefix + 'n_components')),
                       str(get_parameter_range(grid, grid_prefix + 'kernel')),
                       str(get_parameter_range(grid, grid_prefix + 'degree'))))


def get_summary(name, X, y, cv, estimator, grid_scores):
    str = StringIO()
    str.write('Experiment:\t%s\n' % name)

    str.write('Dataset:\t%d samples, %d features, %d classes\n' % (X.shape[0], X.shape[1], len(np.unique(y))))
    # cv = grid_search.cv
    if isinstance(cv, KFoldRepeatStable):
        str.write('CV:\t\t\t%dx%d stratified folds, total - %d, random state - %d\n' %
                  (cv.n_folds, cv.n_reps, len(cv), cv.random_state))
    elif hasattr(cv, '__len__'):
        str.write('CV:\t\t\tcustom with %d splits\n' % (len(cv)))
    else:
        raise NotImplementedError
    str.write('Model:\n')

    grid = map(lambda p: p.parameters, grid_scores)
    # if not isinstance(grid, ParameterGrid):
    #     grid = ParameterGrid(grid)
    get_estimator_description(estimator, grid, outfile=str, print_prefix='\t')

    best_score_params = grid_scores[
        np.argmax(np.array(map(lambda score: score.mean_validation_score, grid_scores)))]
    best_score_mean = best_score_params.mean_validation_score
    best_score_std = np.std(best_score_params.cv_validation_scores)
    str.write('Best score:\t%.3g +- %.3g\n' % (np.round(best_score_mean, 3), np.round(best_score_std, 3)))
    str.write('Best params:\t%s\n' % best_score_params.parameters)

    return str.getvalue()


if __name__ == '__main__':
    from estimators import estimators

    name, estimator, grid = estimators[1]
    get_estimator_description(estimator, ParameterGrid(grid))
