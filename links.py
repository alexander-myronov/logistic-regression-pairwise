from functools import partial
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.base import BaseEstimator


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinksClassifier(BaseEstimator):
    def __init__(self, alpha, gamma=1, kernel='linear', kernel_gamma='auto',
                 percent_pairs=0.5, verbose=False,
                 sampling='random'):
        self.v = None
        self.alpha = alpha
        self.X = None
        self.gamma = gamma
        self.n_classes = None
        self.kernel_dict = {
            'linear': self.linear_kernel,
            'rbf': self.rbf_kernel,
        }
        self.kernel = kernel
        self.kernel_f = self.kernel_dict[kernel]
        self.kernel_gamma = kernel_gamma
        self.percent_pairs = percent_pairs
        self.verbose = verbose

        self.samling_dict = {
            'random': self.sample_pairs_random,
            'max_kdist': self.sample_pairs_max_kdist
        }
        self.sampling = sampling
        self.sampling_f = self.samling_dict[sampling]

    def linear_kernel(self, X, X_prim=None):
        return X

    def rbf_kernel(self, X, X_prim=None):
        if X_prim is None:
            X_prim = self.fullX
        if self.kernel_gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.kernel_gamma
        dist = cdist(X, X_prim, metric='sqeuclidean')
        return np.exp(-gamma * dist)

    def fit(self, X, y):
        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        self.X = X

        self.X1, self.X2, self.z = self.sampling_f(X, y)
        self.n_classes = len(np.unique(y))
        self.fullX = np.vstack([X, self.X1, self.X2])
        self.K = self.kernel_f(self.fullX, self.fullX)

        self.v = np.zeros(shape=(self.n_classes - 1, self.K.shape[1]))

        f = partial(self.loss, X, y, self.X1, self.X2, self.z, alpha=self.alpha, gamma=self.gamma)

        fprime = partial(self.loss_grad, X, y, self.X1, self.X2, self.z,
                         alpha=self.alpha, gamma=self.gamma)

        fprime_lloss = partial(self.links_loss_grad, X, self.X1, self.X2, self.z)
        f_lloss = partial(self.links_loss, self.X1, self.X2, self.z)

        for _ in xrange(0):
            v_test = np.random.normal(size=self.v.ravel().shape)
            num_grad_lloss = scipy.optimize.approx_fprime(
                v_test,
                f_lloss,
                1e-8)
            an_grad_lloss = fprime_lloss(v_test)
            err = np.abs(num_grad_lloss - an_grad_lloss)
            print('lloss gradient error: %f' % np.sum(err ** 2))

            num_grad_all = scipy.optimize.approx_fprime(
                v_test,
                f,
                1e-8)
            an_grad_all = fprime(v_test)
            err = np.abs(num_grad_all - an_grad_all)
            print('total gradient error: %f' % np.sum(err ** 2))

        def cb(v):
            tr_loss, l_loss, n_loss = f(v, split=True)
            print('loss: traditional=%.3f, links=%.3f, penalty=%.3f' % (tr_loss, l_loss, n_loss))

        res = scipy.optimize.fmin_l_bfgs_b(f,
                                           self.v,
                                           approx_grad=True,
                                           fprime=fprime,
                                           maxiter=10000,
                                           disp=0,
                                           callback=cb if self.verbose else None)

        self.v = res[0]
        # res = scipy.optimize.fmin_ncg(f,
        #                               self.v,
        #                               # approx_grad=True,
        #                               fprime=fprime,
        #                               maxiter=10000,
        #                               disp=1,
        #                               full_output=1,
        #                               callback=cb if self.verbose else None)
        #
        # self.v = res[0]
        self.v = self.v.reshape(self.n_classes - 1, -1)
        return self

    def sample_pairs_random(self, X, y):
        # np.random.seed(44)
        num = int(len(y) * self.percent_pairs)

        choice1 = np.random.choice(len(y), size=num, replace=True)
        X1 = X[choice1]
        choice2 = np.random.choice(len(y), size=num, replace=True)
        X2 = X[choice2]
        z = (y[choice1] == y[choice2]).astype(float)

        return X1, X2, z

    def sample_pairs_max_kdist(self, X, y):
        num = int(len(y) * self.percent_pairs)

        probs = LinksClassifier(
            alpha=0,
            gamma=self.gamma,
            kernel=self.kernel,
            kernel_gamma=self.kernel_gamma,
            percent_pairs=self.percent_pairs,
            sampling='random'
        ).fit(X, y).predict_proba(X)[:, 1]
        uncertainty = np.abs(probs - 0.5)
        worst_samples = np.argsort(uncertainty)[:num]

        X1 = X[worst_samples]

        if self.kernel == 'linear':
            K = squareform(pdist(X, metric='euclidean'))
        else:
            K = self.kernel_f(X, X)

        def kdist(i, j):
            return K[i, i] + K[j, j] - 2 * K[i, j]

        worst_conterparts = np.zeros(len(worst_samples), dtype=int)
        for i, sample_index in enumerate(worst_samples):
            counterparts = list(xrange(len(y)))
            counterparts.remove(sample_index)
            # kdistances = [kdist(i, j) for j in counterparts]
            kdistances = map(lambda j: kdist(sample_index, j), counterparts)
            max_j = np.argmax(kdistances)
            worst_conterparts[i] = max_j

        X2 = X[worst_conterparts]
        z = (y[worst_samples] == y[worst_conterparts])
        z = z.astype(int)
        # z[z == 0] = -1

        return X1, X2, z

    def predict_proba(self, X):
        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        return self.predict_proba_(X, self.v)

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)

    def get_exps(self, X, v):
        exps = np.zeros(shape=(len(X), self.n_classes - 1))
        for k in xrange(self.n_classes - 1):
            v_k = v[k]
            exp = np.exp(np.dot(self.kernel_f(X), v_k))
            exps[:, k] = exp

        return exps

    def predict_proba_(self, X, v):
        exps = self.get_exps(X, v)
        denom = 1 + exps.sum(axis=1)
        probs = np.zeros(shape=(len(X), self.n_classes))
        for k in xrange(self.n_classes - 1):
            probs[:, k] = exps[:, k] / denom

        probs[:, -1] = 1 - probs[:, :-1].sum(axis=1)
        return probs

    def loss(self, X, y, X1, X2, z, v, gamma=1, alpha=1, split=False):
        labeled_loss = 0
        v_by_class = v.reshape(self.n_classes - 1, -1)
        probs = self.predict_proba_(X, v_by_class)
        for k in xrange(self.n_classes):
            probs_k = probs[:, k]
            p_k = (y == k).astype(float)
            labeled_loss_k = np.sum((p_k - probs_k) ** 2)
            labeled_loss += labeled_loss_k

        link_loss = self.links_loss(X1, X2, z, v)
        norm_loss = np.dot(v, v)
        if split:
            return labeled_loss, alpha * link_loss, gamma * norm_loss
        return labeled_loss + alpha * link_loss + gamma * norm_loss

    def links_loss(self, X1, X2, z, v):
        v_by_class = v.reshape(self.n_classes - 1, -1)
        probs_i = self.predict_proba_(X1, v_by_class)
        probs_j = self.predict_proba_(X2, v_by_class)

        link_loss_by_class = np.zeros(shape=(len(z), self.n_classes))
        for k in xrange(self.n_classes):
            links_loss_k = probs_i[:, k] * probs_j[:, k]
            link_loss_by_class[:, k] = links_loss_k
        link_loss = np.sum((z - link_loss_by_class.sum(axis=1)) ** 2)
        return link_loss

    def dpkx_dvk(self, exps, k):

        k_index = np.zeros(self.n_classes - 1, dtype=bool)
        k_index[k] = True

        grad = (exps[:, k] * (1 + exps[:, ~k_index].sum(axis=1))) / ((1 + exps.sum(axis=1)) ** 2)
        return grad

    def dpkx_dvm(self, exps, k, m):
        grad = (-exps[:, k] * exps[:, m]) / ((1 + exps.sum(axis=1)) ** 2)
        return grad

    def dpnx_dvm(self, exps, m):
        grad = (-exps[:, m]) / ((1 + exps.sum(axis=1)) ** 2)
        return grad

    def loss_grad(self, X, y, X1, X2, z, v, gamma=1, alpha=1):

        v_by_class = v.reshape(self.n_classes - 1, -1)

        exps = self.get_exps(X, v_by_class)
        probs = self.predict_proba_(X, v_by_class)

        labeled_loss_grad = np.zeros(shape=v_by_class.shape)

        for k in xrange(self.n_classes - 1):
            labeled_loss_grad_k = np.zeros(labeled_loss_grad.shape[1])
            for l in xrange(self.n_classes):
                probs_l = probs[:, l]
                p_l = (y == l).astype(float)

                # g1 = self.dpkx_dvk(exps, k)
                # g2 = self.dpkx_dvm(exps, k)
                # diff = (g1 - g2).sum()
                if k == l:
                    prob_grad = self.dpkx_dvk(exps, k=k)
                else:
                    if l == self.n_classes - 1:
                        prob_grad = self.dpnx_dvm(exps, m=k)
                    else:
                        prob_grad = self.dpkx_dvm(exps, k=l, m=k)

                prob_grad = self.K[:len(X)] * prob_grad.reshape(-1, 1)
                k_l_grad = prob_grad * (p_l - probs_l).reshape(-1, 1)

                labeled_loss_grad_k += k_l_grad.sum(axis=0)
            labeled_loss_grad[k] = labeled_loss_grad_k

        links_loss_grad = self.links_loss_grad(X, X1, X2, z, v)

        return - 2 * labeled_loss_grad.ravel() \
               + alpha * links_loss_grad \
               + 2 * gamma * v_by_class.ravel()

    def links_loss_grad(self, X, X1, X2, z, v):
        v_by_class = v.reshape(self.n_classes - 1, -1)
        exps_i = self.get_exps(X1, v_by_class)
        probs_i = self.predict_proba_(X1, v_by_class)
        exps_j = self.get_exps(X2, v_by_class)
        probs_j = self.predict_proba_(X2, v_by_class)

        links_loss_grad = np.zeros(shape=v_by_class.shape)
        for k in xrange(self.n_classes - 1):

            link_loss_grad_by_class_sum1 = np.zeros(shape=(len(z), self.n_classes))
            link_loss_grad_by_class_sum2 = np.zeros(shape=(
                len(z),
                self.n_classes,
                links_loss_grad.shape[1]))

            for l in xrange(self.n_classes):
                links_loss_grad_k = probs_i[:, l] * probs_j[:, l]
                link_loss_grad_by_class_sum1[:, l] = links_loss_grad_k

                if k == l:
                    prob_grad_i = self.dpkx_dvk(exps_i, k=k)
                    prob_grad_j = self.dpkx_dvk(exps_j, k=k)
                else:
                    if l == self.n_classes - 1:
                        prob_grad_i = self.dpnx_dvm(exps_i, m=k)
                        prob_grad_j = self.dpnx_dvm(exps_j, m=k)
                    else:
                        prob_grad_i = self.dpkx_dvm(exps_i, k=l, m=k)
                        prob_grad_j = self.dpkx_dvm(exps_j, k=l, m=k)

                prob_grad_i = self.K[len(X):len(X1) + len(X)] * prob_grad_i.reshape(-1, 1)
                prob_grad_j = self.K[len(X1) + len(X):] * prob_grad_j.reshape(-1, 1)

                p2_grad = probs_j[:, l].reshape(-1, 1) * prob_grad_i + \
                          probs_i[:, l].reshape(-1, 1) * prob_grad_j
                link_loss_grad_by_class_sum2[:, l] = p2_grad
            grad = (z - link_loss_grad_by_class_sum1.sum(axis=1)).reshape(-1, 1) * \
                   link_loss_grad_by_class_sum2.sum(axis=1)
            links_loss_grad[k] = np.sum(grad, axis=0)
        return -2 * links_loss_grad.ravel()