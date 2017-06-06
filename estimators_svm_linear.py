from functools import partial

from sklearn.svm import SVC

from links import LinksClassifier
from links_vs_npklr_vs_svm import estimator_tuple
from logit import LogisticRegressionPairwise
from grid import *

estimators = [
    # estimator_tuple(
    #     name='Links(linear)',
    #     estimator=LinksClassifier(kernel='linear', sampling='predefined', solver='tnc',
    #                               init='normal_univariate'),
    #     kwargs_func=lambda kw: kw,
    #     grid_func=partial(links_grid_linear_no_labels, method=4)),
    # estimator_tuple(
    #     name='Links(linear,joint loss)',
    #     estimator=LinksClassifier(kernel='linear', beta=None, sampling='predefined', solver='tnc'),
    #     kwargs_func=lambda kw: kw,
    #     grid_func=lambda *a, **kw: pop_kwargs(partial(links_grid_linear, method=4)(*a, **kw),
    #                                           to_pop=['beta'])),
    # estimator_tuple(
    #     name='NPKLR(linear)',
    #     estimator=LogisticRegressionPairwise(kernel='linear', sampling='predefined'),
    #     kwargs_func=labels_links,
    #     grid_func=partial(links_grid_linear_no_labels, method=4)),
    estimator_tuple(
        name='SVM(linear)',
        estimator=SVC(kernel='linear', cache_size=1024),
        kwargs_func=labels_only,
        grid_func=partial(svm_grid_linear, method=4)),
]
