from functools import partial

from sklearn.svm import SVC

from links import LinksClassifier
from links_vs_npklr_vs_svm import estimator_tuple
from logit import LogisticRegressionPairwise
from grid import *
from sklearn.multiclass import OneVsOneClassifier

estimators = [
    estimator_tuple(
        name='Links(linear)',
        estimator=LinksClassifier(kernel='linear', sampling='predefined', solver='tnc',
                                  init='normal_univariate'),
        kwargs_func=lambda kw: kw,
        grid_func=partial(links_grid_linear_no_labels, method=4)),

    estimator_tuple(
        name='NPKLR(linear)',
        estimator=OneVsOneClassifier(LogisticRegressionPairwise(kernel='linear',
                                                                sampling='predefined')),
        kwargs_func=labels_links,
        grid_func=prepend(partial(links_grid_linear_no_labels, method=4), 'estimator__')),
    estimator_tuple(
        name='Links(rbf)',
        estimator=LinksClassifier(kernel='rbf', sampling='predefined', solver='tnc',
                                  init='normal_univariate'),
        kwargs_func=lambda kw: kw,
        grid_func=partial(links_grid_rbf_no_labels, method=4)),

    estimator_tuple(
        name='NPKLR(rbf)',
        estimator=OneVsOneClassifier(LogisticRegressionPairwise(kernel='rbf',
                                                                sampling='predefined')),
        kwargs_func=labels_links,
        grid_func=prepend(partial(links_grid_rbf_no_labels, method=4), 'estimator__')),

]
