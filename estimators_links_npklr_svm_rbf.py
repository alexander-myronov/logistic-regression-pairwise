from functools import partial

from sklearn.svm import SVC

from grid import *
from links import LinksClassifier
from links_vs_npklr_vs_svm import estimator_tuple
from logit import LogisticRegressionPairwise

estimators = [
    estimator_tuple(
        name='Links(rbf)',
        estimator=LinksClassifier(kernel='rbf', sampling='predefined', solver='tnc'),
        kwargs_func=lambda kw: kw,
        grid_func=partial(links_grid_rbf, method=4)),
    estimator_tuple(
        name='Links(rbf,joint loss)',
        estimator=LinksClassifier(kernel='linear', beta=None, sampling='predefined', solver='tnc'),
        kwargs_func=lambda kw: kw,
        grid_func=lambda *a, **kw: pop_kwargs(partial(links_grid_linear, method=4)(*a, **kw),
                                              to_pop=['beta'])),
    estimator_tuple(
        name='NPKLR(rbf)',
        estimator=LogisticRegressionPairwise(kernel='rbf', sampling='predefined'),
        kwargs_func=labels_links,
        grid_func=partial(links_grid_rbf, method=4)),
    estimator_tuple(
        name='SVM(rbf)',
        estimator=SVC(kernel='rbf', cache_size=1024),
        kwargs_func=labels_only,
        grid_func=partial(svm_grid_rbf, method=4)),
]
