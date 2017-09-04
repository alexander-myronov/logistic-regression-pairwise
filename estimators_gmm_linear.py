from functools import partial

from sklearn.svm import SVC

from gmm_with_links import GmmWithLinks
from links import LinksClassifier
from links_vs_npklr_vs_svm import estimator_tuple
from logit import LogisticRegressionPairwise
from grid import *

estimators = [
    estimator_tuple(
        name='GMM(linear)',
        estimator=GmmWithLinks(delta=1e-10, verbose=False),
        kwargs_func=lambda kw: kw,
        grid_func=partial(gmm_grid_linear, method=4)),
]
