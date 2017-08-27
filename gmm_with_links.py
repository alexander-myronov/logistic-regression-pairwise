from __future__ import division, print_function

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score


from pymix.distributions.multinormal import MultiNormalDistribution
from pymix.distributions.normal import NormalDistribution

from pymix.models.constrained import ConstrainedMixtureModel
from pymix.util.constrained_dataset import ConstrainedDataSet
import numpy as np


from pymix.util.dataset import DataSet
from start_sensitivity import split_dataset


class GmmWithLinks(BaseEstimator):
    def __init__(self, delta=1e-8,
                 positive_prior=1,
                 negative_prior=1,
                 max_iter=100,
                 max_run=1,
                 verbose=False):
        self.delta = delta
        self.positive_prior = positive_prior
        self.negative_prior = negative_prior
        self.max_iter = max_iter
        self.cm = None
        self.verbose = verbose
        self.max_run = max_run

    def fit(self, X, y=None, **kwargs):
        assert 'X1' in kwargs and 'X2' in kwargs and 'z' in kwargs
        X1 = kwargs['X1']
        X2 = kwargs['X2']
        z = kwargs['z']
        Xu = kwargs['Xu']

        if y is not None and len(y) > 0:
            n_classes = len(np.unique(y))
        else:
            n_classes = kwargs['n_classes']

        X_in = np.vstack([X1, X2, Xu])
        positive_constraints = np.zeros(shape=(len(X_in), len(X_in)))
        negative_constraints = np.zeros(shape=(len(X_in), len(X_in)))

        for i_z, z_value in enumerate(z):
            if z_value == 1:
                positive_constraints[i_z, i_z + len(z)] = \
                    positive_constraints[i_z + len(z), i_z] = 1
            elif z_value == 0:
                negative_constraints[i_z, i_z + len(z)] = \
                    negative_constraints[i_z + len(z), i_z] = 1
            else:
                raise Exception('unexpected value in pair vector z')

        ds = ConstrainedDataSet().fromArray(X_in.tolist())
        ds.setPairwiseConstraints(positive_constraints, negative_constraints)

        priors = [MultiNormalDistribution([0] * X_in.shape[1], np.eye(X_in.shape[1])) \
                  for _ in xrange(n_classes)]
        prior_probs = [1.0 / n_classes] * n_classes
        self.cm = ConstrainedMixtureModel(n_classes, prior_probs, priors)
        # prev_pos = self.cm.modelInitialization(ds,
        #                                        self.positive_importance,
        #                                        self.negative_importance,
        #                                        prior_type=3,
        #                                        rtype=0)
        # post, lik = self.cm.EM(ds,
        #                        self.max_iter,
        #                        self.delta,
        #                        self.positive_importance,
        #                        self.negative_importance,
        #                        previous_posterior=prev_pos,
        #                        prior_type=3,
        #                        silent=False)
        post, lik = self.cm.randMaxEM(ds,
                                      self.max_run,
                                      self.max_iter,
                                      self.delta,
                                      self.positive_prior,
                                      self.negative_prior,
                                      prior_type=3,
                                      silent=not self.verbose)
        if self.verbose:
            print('Log-likelihood: %f' % lik)
        return self

    def predict(self, X):
        ds_test = ConstrainedDataSet().fromArray(X.tolist())
        ds_test.setPairwiseConstraints(np.zeros(shape=(len(X), len(X))),
                                       np.zeros(shape=(len(X), len(X))))
        # prev_pos = self.cm.modelInitialization(ds_test,
        #                                        self.positive_importance,
        #                                        self.negative_importance,
        #                                        3,
        #                                        rtype=0)
        y_pred = self.cm.classify(ds_test,
                                  self.positive_prior,
                                  self.negative_prior,
                                  np.full((2, len(X)), fill_value=np.nan), 3, silent=True)
        return y_pred


if __name__ == '__main__':
    from links_vs_npklr_vs_svm import load_ds
    from sklearn.metrics import adjusted_rand_score
    X, y = load_ds('data/breast-cancer_scale.libsvm')

    train, test = next(StratifiedShuffleSplit(n_splits=1,
                                              test_size=0.8,
                                              random_state=43).split(X, y))
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    labels_index, choice1, choice2, z_tr, unsup_choice = split_dataset(
        X_train, y_train,
        percent_labels=0.1,
        percent_links=0.4,
        percent_unlabeled=0.4,
        labels_and_links_separation_degree=1,
        return_index=True,
        random_state=42)

    X_tr = X_train[labels_index]
    y_tr = y_train[labels_index]
    X1_tr = X_train[choice1]
    X2_tr = X_train[choice2]
    Xu_tr = X_train[unsup_choice]

    true_training_labels = np.concatenate([y_train[choice1],
                                           y_train[choice2],
                                           y_train[unsup_choice]])

    scores = []
    for _ in xrange(25):
        estimator = GmmWithLinks(1e-12,
                                 0.5,
                                 0.5,
                                 max_run=1,
                                 verbose=False).fit(X_tr, y_tr,
                                                    X1=X1_tr,
                                                    X2=X2_tr,
                                                    z=z_tr,
                                                    Xu=Xu_tr)
        y_pred_test = estimator.predict(X_test)
        score = adjusted_rand_score(y_test, y_pred_test)
        print(score)
        scores.append(score)

    print(np.mean(scores), np.max(scores))
