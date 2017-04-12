from __future__ import division, print_function, with_statement
from functools import partial
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.base import BaseEstimator
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinksClassifier(BaseEstimator):
    def __init__(self,
                 alpha=1,
                 beta=1,
                 kernel='linear',
                 gamma='auto',
                 percent_pairs=0.5,
                 verbose=False,
                 sampling='random',
                 init='zeros',
                 delta=1,
                 solver='tnc'):
        self.v = None
        self.alpha = alpha
        self.X = None
        self.beta = beta
        self.n_classes = None
        self.delta = delta
        self.kernel = kernel

        self.gamma = gamma
        self.percent_pairs = percent_pairs
        self.verbose = verbose

        self.sampling = sampling
        self.init = init
        self.solver = solver

    def kernel_f(self, X, X_prim=None):
        if self.kernel == 'linear':
            return self.linear_kernel(X, X_prim)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, X_prim)

    def linear_kernel(self, X, X_prim=None):
        return X

    def rbf_kernel(self, X, X_prim=None):
        if X_prim is None:
            X_prim = self.fullX
        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
        dist = cdist(X, X_prim, metric='sqeuclidean')
        return np.exp(-gamma * dist)

    def init_f(self):
        if self.init == 'zeros':
            return np.zeros(shape=(self.n_classes - 1, self.K.shape[1]))
        if self.init == 'normal':
            return np.random.normal(loc=0, scale=0.5,
                                    size=(self.n_classes - 1, self.K.shape[1]))
        if self.init == 'normal_univariate':
            means = np.mean(self.K, axis=0)
            stds = np.std(self.K, axis=0)
            return np.random.normal(loc=means, scale=stds,
                                    size=(self.n_classes - 1, self.K.shape[1]))
        if self.init == 'normal_multivariate':
            means = np.mean(self.K, axis=0)
            cov = np.cov(self.K, rowvar=False)
            return np.random.multivariate_normal(mean=means, cov=cov,
                                                 size=(self.n_classes - 1))
        if self.init == 'random_labels':
            result = np.zeros(shape=(self.n_classes - 1, self.K.shape[1]))
            for k in xrange(self.n_classes - 1):
                rand = np.random.choice(np.where(self.y == k)[0], size=1)
                result[k, :] = self.K[:len(self.X)][rand]
            return result
        if self.init == 'random_links_diff':
            result = np.zeros(shape=(self.n_classes - 1, self.K.shape[1]))
            for k in xrange(self.n_classes - 1):
                rand = np.random.choice(np.where(self.z == 0)[0], size=1)
                result[k, :] = \
                    (self.K[len(self.X):len(self.X) + len(self.X1)][rand] -
                     self.K[len(self.X) + len(self.X1):len(self.X) + len(self.X1) + len(self.X2)]
                     [rand])
            return result
        else:
            raise Exception("wrong init method: %s" % self.init)

    def fit(self, X, y, **kwargs):
        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        self.X = X
        self.n_classes = len(np.unique(y))
        assert np.sum(y == -1) == 0
        self.y = np.copy(y)

        X1, X2, z, Xu = self.sampling_f(X, y, **kwargs)
        self.X1, self.X2, self.z, self.Xu = self.preprocess_additional_arrays(X, y, X1, X2, z, Xu)

        self.fullX = np.vstack([X, self.X1, self.X2, self.Xu])
        self.K = self.kernel_f(self.fullX, self.fullX)

        self.v = self.init_f()

        f = partial(self.loss, X, y, self.X1, self.X2, self.z, self.Xu,
                    alpha=self.alpha, beta=self.beta, delta=self.delta)

        fprime = partial(self.loss_grad, X, y, self.X1, self.X2, self.z, self.Xu,
                         alpha=self.alpha, beta=self.beta, delta=self.delta)

        for _ in xrange(kwargs.pop('n_gradient_checks', 0)):
            v_test = np.random.normal(size=self.v.ravel().shape)
            # num_grad_lloss = scipy.optimize.approx_fprime(
            #     v_test,
            #     f_lloss,
            #     1e-8)
            # an_grad_lloss = fprime_lloss(v_test)
            # err = np.abs(num_grad_lloss - an_grad_lloss)
            # print('lloss gradient error: %f' % np.sum(err ** 2))

            num_grad_all = scipy.optimize.approx_fprime(
                v_test,
                f,
                1e-8)
            an_grad_all = fprime(v_test)
            err = np.abs(num_grad_all - an_grad_all)
            print('total gradient error: %f' % np.sum(err ** 2))

        start_time = time.time()
        result_v = self.v

        def cb(v):
            global result_v
            tr_loss, l_loss, n_loss, u_loss = f(v, split=True)
            if self.verbose:
                print('loss: traditional=%.3f, links=%.3f, norm=%.3f, unsup=%.3f' % \
                      (tr_loss, l_loss, n_loss, u_loss))
            if time.time() - start_time > 80:
                result_v = v
                raise Exception('timeout')

        if self.verbose:
            print('K size: ', self.K.shape[0])
        try:

            if self.solver == 'ncg':
                res = scipy.optimize.fmin_ncg(f,
                                              self.v,
                                              # approx_grad=True,
                                              fprime=fprime,
                                              maxiter=100,
                                              disp=0,
                                              callback=cb,
                                              avextol=0.001,
                                              epsilon=1e-5)
            elif self.solver == 'tnc':
                res = scipy.optimize.fmin_tnc(f,
                                              self.v,
                                              # approx_grad=True,
                                              fprime=fprime,
                                              maxfun=500,
                                              ftol=1e-4,
                                              disp=0,
                                              callback=cb,
                                              # avextol=0.001,
                                              )
                res = res[0]

            elif self.solver == 'bfgs':
                res = scipy.optimize.fmin_bfgs(f,
                                               self.v,
                                               # approx_grad=True,
                                               fprime=fprime,
                                               maxiter=1000,
                                               disp=0,
                                               callback=cb,
                                               # pgtol=0.001,
                                               )
            else:
                raise Exception('unknown solver: %s' % self.solver)
            self.v = res
        except Exception:
            self.v = result_v.ravel()
            print('Timeout')

        self.last_loss = f(self.v)
        if self.verbose:
            print(self.gamma, 'training stopped at ', self.last_loss)

        self.v = self.v.reshape(self.n_classes - 1, -1)
        return self

    def calc_loss(self, X, y, X1, X2, z, Xu):
        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        X1, X2, z, Xu = self.preprocess_additional_arrays(X, y, X1, X2, z, Xu)
        fullX = np.vstack([X, X1, X2, Xu])
        assert fullX.shape[1] == self.fullX.shape[1]
        # K = self.kernel_f(fullX, self.fullX)
        v = self.v.ravel()
        return self.loss(X, y, X1, X2, z, Xu, v,
                         alpha=self.alpha,
                         beta=self.beta,
                         delta=self.delta,
                         split=True)

    def preprocess_additional_arrays(self, X, y, X1, X2, z, Xu):
        if X1.shape[1] == X.shape[1] - 1:
            X1 = np.hstack([np.ones(shape=(X1.shape[0], 1)), X1])
        if X2.shape[1] == X.shape[1] - 1:
            X2 = np.hstack([np.ones(shape=(X2.shape[0], 1)), X2])
        if Xu.shape[1] == X.shape[1] - 1:
            Xu = np.hstack([np.ones(shape=(Xu.shape[0], 1)), Xu])
        return X1, X2, z, Xu

    def sampling_f(self, X, y, **kwargs):
        if self.sampling == 'random':
            X1, X2, z = self.sample_pairs_random(X, y)
            Xu = np.zeros(shape=(0, X.shape[1]))
        elif self.sampling == 'max_kdist':
            X1, X2, z = self.sample_pairs_max_kdist(X, y)
            Xu = np.zeros(shape=(0, X.shape[1]))
        elif self.sampling == 'predefined':
            X1 = kwargs.get('X1', np.zeros(shape=(0, X.shape[1])))
            X2 = kwargs.get('X2', np.zeros(shape=(0, X.shape[1])))
            z = kwargs.get('z', np.zeros(0))
            Xu = kwargs.get('Xu', np.zeros(shape=(0, X.shape[1])))

        return X1, X2, z, Xu

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
            beta=self.beta,
            kernel=self.kernel,
            gamma=self.gamma,
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

    def loss(self, X, y, X1, X2, z, Xu, v, alpha=1, beta=1, delta=1, split=False):
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
        u_loss = self.unsup_loss(Xu, v)

        if split:
            return alpha / max(len(y), 1) * labeled_loss, \
                   beta / max(len(z), 1) * link_loss, \
                   delta / max(len(Xu), 1) * u_loss, \
                   norm_loss
        return alpha / max(len(y), 1) * labeled_loss \
               + beta / max(len(z), 1) * link_loss \
               + delta / max(len(Xu), 1) * u_loss \
               + norm_loss

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

    def unsup_loss(self, Xu, v):
        v_by_class = v.reshape(self.n_classes - 1, -1)
        probs = self.predict_proba_(Xu, v_by_class)
        unsup_loss_by_class = np.zeros(shape=(len(Xu), self.n_classes))
        for k in xrange(self.n_classes):
            links_loss_k = -(probs[:, k] ** 2)
            unsup_loss_by_class[:, k] = links_loss_k
        u_loss = np.sum(unsup_loss_by_class, axis=1)
        if len(u_loss) == 0:
            return 0
        return np.sum(u_loss)

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

    def loss_grad(self, X, y, X1, X2, z, Xu, v, alpha=1, beta=1, delta=1):

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
        u_loss_grad = self.unsup_loss_grad(X, X1, X2, Xu, v)

        return - 2 * alpha / max(len(y), 1) * labeled_loss_grad.ravel() \
               - 2 * beta / max(len(z), 1) * links_loss_grad \
               - 2 * delta / max(len(Xu), 1) * u_loss_grad \
               + 2 * v_by_class.ravel()

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
                prob_grad_j = self.K[len(X1) + len(X):len(X) + len(X1) + len(X2)] * \
                              prob_grad_j.reshape(-1, 1)

                p2_grad = probs_j[:, l].reshape(-1, 1) * prob_grad_i + \
                          probs_i[:, l].reshape(-1, 1) * prob_grad_j
                link_loss_grad_by_class_sum2[:, l] = p2_grad
            grad = (z - link_loss_grad_by_class_sum1.sum(axis=1)).reshape(-1, 1) * \
                   link_loss_grad_by_class_sum2.sum(axis=1)
            links_loss_grad[k] = np.sum(grad, axis=0)
        return links_loss_grad.ravel()

    def unsup_loss_grad(self, X, X1, X2, Xu, v):
        v_by_class = v.reshape(self.n_classes - 1, -1)

        exps = self.get_exps(Xu, v_by_class)
        probs = self.predict_proba_(Xu, v_by_class)

        u_loss_grad = np.zeros(shape=v_by_class.shape)

        for k in xrange(self.n_classes - 1):
            u_loss_grad_k = np.zeros(u_loss_grad.shape[1])
            for l in xrange(self.n_classes):
                probs_l = probs[:, l]

                if k == l:
                    prob_grad = self.dpkx_dvk(exps, k=k)
                else:
                    if l == self.n_classes - 1:
                        prob_grad = self.dpnx_dvm(exps, m=k)
                    else:
                        prob_grad = self.dpkx_dvm(exps, k=l, m=k)

                prob_grad = \
                    self.K[len(X) + len(X1) + len(X2):len(X) + len(X1) + len(X2) + len(Xu)] * \
                    prob_grad.reshape(-1, 1)
                k_l_grad = prob_grad * probs_l.reshape(-1, 1)

                u_loss_grad_k += k_l_grad.sum(axis=0)
            u_loss_grad[k] = u_loss_grad_k
        return u_loss_grad.ravel()
