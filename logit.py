from collections import OrderedDict
from functools import partial
from warnings import warn
from matplotlib.colors import ListedColormap
import numpy as np
import scipy.optimize
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator

from sklearn.datasets import load_svmlight_file, make_moons, make_circles, \
    make_multilabel_classification, make_classification
from sklearn.metrics import roc_auc_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, learning_curve, \
    train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.svm import SVC
from links import LinksClassifier
from plot import plot_2d_slice


class LogisticRegression(BaseEstimator):
    def __init__(self, alpha, kernel='linear', gamma='auto'):
        self.beta = None
        self.alpha = alpha
        self.X = None
        self.gamma = gamma
        self.kernel = kernel
        # self.kernel_f = self.kernel_dict[kernel]

    def kernel_f(self, X):
        if self.kernel == 'linear':
            return self.linear_kernel(X)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X)

    def rbf_kernel(self, X):
        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
        dist = cdist(X, self.X, metric='sqeuclidean')
        return np.exp(-gamma * dist)

    def linear_kernel(self, X):
        return X

    def fit(self, X, y, **kwargs):

        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        self.X = X

        X_k = self.kernel_f(X)
        self.beta = np.zeros(X_k.shape[1])

        y = np.array(y)
        y[y == 0] = -1

        f = partial(self.loss, X, y, alpha=self.alpha)

        fprime = partial(self.loss_grad, X, y, alpha=self.alpha)

        # for _ in xrange(10):
        #     err = scipy.optimize.check_grad(f, fprime, np.random.normal(size=self.beta.shape))
        #     print('gradient error: %f' % err)

        # self.beta, _, _ = scipy.optimize.fmin_l_bfgs_b(f,
        #                                           self.beta,
        #                                           # approx_grad=True,
        #                                           fprime=fprime,
        #                                          # maxfun=10000,
        #                                           maxiter=10000,
        #                                           disp=1)

        try:
            res = scipy.optimize.fmin_tnc(f,
                                          self.beta,
                                          # approx_grad=True,
                                          fprime=fprime,
                                          # maxfun=10000,
                                          maxfun=150,
                                          disp=0,
                                          ftol=1e-4)
            res = res[0]
        except:
            res = self.beta
        # last_loss = f(res)
        self.beta = res

        return self

    def predict(self, X):
        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        probs = self.predict_proba_(X, self.beta)
        return np.round(probs, 0)

    def predict_proba(self, X):
        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        probs = self.predict_proba_(X, self.beta)
        probs = self.sigmoid(probs)
        return np.vstack([1 - probs, probs]).T

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict_proba_(self, X, beta):
        # return LogisticRegression.sigmoid(np.dot(X, beta))
        return LogisticRegression.sigmoid(np.dot(self.kernel_f(X), beta))
        # return np.dot(self.kernel_f(X), beta)

    def loss(self, X, y, beta, alpha=0):
        K = self.kernel_f(X)
        the_loss = np.log(1 / (1 + np.exp(-y * np.dot(K, beta))))
        the_loss = np.mean(the_loss)
        norm = np.dot(beta.T, beta)
        return -the_loss + alpha * norm

    def loss_grad(self, X, y, beta, alpha=0):
        v = self.predict_proba_(X, beta)
        K = self.kernel_f(X)
        the_loss_grad = (np.exp(-y * v) * y) / (1 + np.exp(-y * v))
        # the_loss_grad = self.sigmoid(v) - y
        # the_loss_grad = the_loss_grad.mean()
        # the_loss_grad = np.dot(the_loss_grad.reshape(-1, 1), beta.reshape(1, -1))
        the_loss_grad = K * the_loss_grad.reshape(-1, 1)
        the_loss_grad = the_loss_grad.mean(axis=0)
        return -the_loss_grad + 2 * alpha * beta

    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'kernel': self.kernel,
            'gamma': self.gamma,
        }

    def set_params(self, **params):
        self.alpha = params.pop('alpha', 1)
        self.gamma = params.pop('gamma', 'auto')
        self.kernel = params.pop('kernel', 'linear')
        return self


class LogisticRegressionPairwise(BaseEstimator):
    def __init__(self, alpha=1, mu=1, percent_pairs=0.5, kernel='linear', gamma='auto',
                 verbose=False):
        self.beta = None
        self.alpha = alpha
        self.mu = mu
        self.percent_pairs = percent_pairs
        self.gamma = gamma
        self.kernel = kernel

        self.verbose = verbose

    def kernel_f(self, X, X_prim):
        if self.kernel == 'linear':
            return self.linear_kernel(X, X_prim)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, X_prim)

    def linear_kernel(self, X, X_prim):
        return X

    def rbf_kernel(self, X, X_prim):
        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
        dist = cdist(X, X_prim, metric='sqeuclidean')
        return np.exp(-gamma * dist)

    def fit(self, X, y):

        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        y = np.array(y)
        y[y == 0] = -1

        X1, X2, z = self.sample_pairs_max_kdist(X, y)
        self.X1 = X1
        self.X2 = X2
        self.X = X

        X_k = self.kernel_f(X, X)
        X1_k = self.kernel_f(X1, X1)
        X2_k = self.kernel_f(X2, X2)
        self.beta = np.zeros(X_k.shape[1])
        if self.kernel != 'linear':
            self.beta1 = np.zeros(X1_k.shape[1])
            self.beta2 = np.zeros(X2_k.shape[1])
        else:
            self.beta1 = np.zeros(shape=0)
            self.beta2 = np.zeros(shape=0)

        self.fullX = np.vstack([X, X1, X2])
        self.K = self.kernel_f(self.fullX, self.fullX)

        # EM
        prev_loss = 0
        loss = 0
        n_iter = 25
        full_beta = np.concatenate([self.beta, self.beta1, self.beta2])

        if self.kernel == 'linear':
            estep_f = self.estep
            loss_f = self.loss_split_other
        else:
            estep_f = self.estep
            loss_f = self.loss_split_other
        while n_iter > 0:

            prev_loss = loss

            E_z1, E_z2 = estep_f(X1, X2, z, full_beta)

            f = partial(loss_f, X, y, X1, X2, z, E_z1, E_z2,
                        alpha=self.alpha,
                        mu=self.mu)

            fprime = partial(self.loss_split_grad, X, y, X1, X2, z, E_z1, E_z2,
                             alpha=self.alpha,
                             mu=self.mu)

            # if self.verbose:
            #     err = scipy.optimize.check_grad(f, fprime, full_beta)
            #     print('gradient error: %f' % err)

            opt = scipy.optimize.fmin_ncg(f,
                                          full_beta,
                                          avextol=1e-4,
                                          fprime=fprime,
                                          # maxfun=10000,
                                          maxiter=1000,
                                          # maxls=100,
                                          full_output=1,
                                          disp=0
                                          )
            full_beta = opt[0]
            loss = opt[1]
            if np.abs(prev_loss - loss) < 1e-5:
                break

            n_iter -= 1
            if self.verbose:
                print(n_iter, loss)
        if n_iter == 0 and np.abs(prev_loss - loss) >= 1e-5:
            # warn
            pass
        self.beta, self.beta1, self.beta2 = self.split_beta(full_beta)
        return self

    def estep(self, X1, X2, z, full_beta):
        E_z1 = 1 / (1 + np.exp(self.predict_proba_(X1, full_beta))) * \
               1 / (1 + np.exp(z * self.predict_proba_(X2, full_beta)))

        E_z2 = 1 / (1 + np.exp(-self.predict_proba_(X1, full_beta))) * \
               1 / (1 + np.exp(-z * self.predict_proba_(X2, full_beta)))
        return E_z1, E_z2

    def estep_split(self, X1, X2, z, full_beta):
        _, beta1, beta2 = self.split_beta(full_beta)
        E_z1 = 1 / (1 + np.exp(self.predict_proba_(X1, beta1, X_prim=X1))) * \
               1 / (1 + np.exp(z * self.predict_proba_(X2, beta2, X_prim=X2)))

        E_z2 = 1 / (1 + np.exp(-self.predict_proba_(X1, beta1, X_prim=X1))) * \
               1 / (1 + np.exp(-z * self.predict_proba_(X2, beta2, X_prim=X2)))
        return E_z1, E_z2

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.round(probs[:, 1], 0).astype(int)

    def predict_proba(self, X):
        full_beta = np.concatenate([self.beta, self.beta1, self.beta2])
        X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        probs = self.predict_proba_(X, full_beta)
        probs = self.sigmoid(probs)
        return np.vstack([1 - probs, probs]).T

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict_proba_(self, X, full_beta, X_prim=None):
        if X_prim is None:
            X_prim = self.fullX
        # full_beta = np.concatenate([self.beta, self.beta1, self.beta2])
        # return LogisticRegression.sigmoid(np.dot(X, beta))
        return np.dot(self.kernel_f(X, X_prim), full_beta)

    def sample_pairs_max_kdist(self, X, y):
        num = int(len(y) * self.percent_pairs)

        probs = LogisticRegression(self.alpha, self.kernel, self.gamma). \
                    fit(X, y).predict_proba(X)[:, 1]
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
        z[z == 0] = -1

        return X1, X2, z

    def sample_pairs_random(self, X, y):
        np.random.seed(44)
        num = int(len(y) * self.percent_pairs)

        y_pos = np.where(y == 1)[0]

        X1_pos_index = np.random.choice(y_pos, size=num, replace=True)
        X2_pos_index = np.random.choice(y_pos, size=num, replace=True)
        z_pos = np.ones(num)

        y_neg = np.where(y == -1)[0]

        X1_neg_index = np.random.choice(y_neg, size=num, replace=True)
        X2_neg_index = np.random.choice(y_neg, size=num, replace=True)
        z_neg = -np.ones(num)

        X1 = np.vstack([X[X1_pos_index], X[X1_neg_index]])
        X2 = np.vstack([X[X2_pos_index], X[X2_neg_index]])
        z = np.concatenate([z_pos, z_neg])

        return X1, X2, z

    def split_beta(self, full_beta):
        beta, beta1, beta2 = np.split(full_beta, [len(self.beta), len(self.beta) + len(self.beta1)])
        return beta, beta1, beta2

    def loss_split_other(self, X, y, X1, X2, z, E_z1, E_z2, full_beta, alpha=1, mu=1):
        # full_X = self.fullX
        # K = self.kernel_f(full_X, full_X)

        # Kp, Kl1, Kl2 = np.split(self.K, [len(X), len(X) + len(X1)])
        Kp = self.K[:len(X), :]
        Kl1 = self.K[len(X):len(X) + len(X1), :]
        Kl2 = self.K[len(X) + len(X1):, :]

        lloss = -np.log(1 + np.exp(-y * np.dot(Kp, full_beta)))
        lloss = lloss.mean()
        ploss = E_z1 * -1 * np.log((1 + np.exp(np.dot(Kl1, full_beta))) * \
                                   (1 + np.exp(z * np.dot(Kl2, full_beta)))) + \
                E_z2 * -1 * np.log((1 + np.exp(-np.dot(Kl1, full_beta))) * \
                                   (1 + np.exp(-z * np.dot(Kl2, full_beta))))
        norm_loss = -np.dot(full_beta.T, full_beta)
        ploss = ploss.mean()
        return - (lloss + mu * ploss + alpha * norm_loss)

    def loss_split_grad(self, X, y, X1, X2, z, E_z1, E_z2, full_beta, alpha=1, mu=1):
        # full_X = self.fullX
        # K = self.kernel_f(full_X, full_X)

        # Kp, Kl1, Kl2 = np.split(self.K, [len(X), len(X) + len(X1)])
        Kp = self.K[:len(X), :]
        Kl1 = self.K[len(X):len(X) + len(X1), :]
        Kl2 = self.K[len(X) + len(X1):, :]

        Vp = np.dot(Kp, full_beta)
        lloss_grad = (np.exp(-y * Vp) * y) / (1 + np.exp(-y * Vp))
        lloss_grad = Kp * lloss_grad.reshape(-1, 1)
        lloss_grad = lloss_grad.mean(axis=0)

        kl1_beta = np.dot(Kl1, full_beta)
        kl2_beta = np.dot(Kl2, full_beta)

        ploss_grad = np.zeros(shape=(len(X1), len(full_beta)))
        for bi in xrange(len(full_beta)):
            kl1_prim = Kl1[:, bi]
            kl2_prim = Kl2[:, bi]
            ploss_grad_i = E_z1 * -1 * 1 / ((1 + np.exp(kl1_beta)) * (1 + np.exp(z * kl2_beta))) * \
                           ((np.exp(kl1_beta) * kl1_prim) * (1 + np.exp(z * kl2_beta)) +
                            (np.exp(z * kl2_beta) * z * kl2_prim) * (1 + np.exp(kl1_beta))) + \
                           E_z2 * -1 * 1 / ((1 + np.exp(-kl1_beta)) * (1 + np.exp(-z * kl2_beta))) * \
                           ((np.exp(-kl1_beta) * -kl1_prim) * (1 + np.exp(-z * kl2_beta)) +
                            (np.exp(-z * kl2_beta) * z * -kl2_prim) * (1 + np.exp(-kl1_beta)))
            ploss_grad[:, bi] = ploss_grad_i
        ploss_grad = ploss_grad.mean(axis=0)
        # ploss = E_z1 * -1 * np.log((1 +) * \
        #                            (1 + np.exp(z * np.dot(Kl2, full_beta)))) + \
        #         E_z2 * -1 * np.log((1 + np.exp(-np.dot(Kl1, full_beta))) * \
        #                            (1 + np.exp(-z * np.dot(Kl2, full_beta))))
        norm_loss_grad = -2 * full_beta

        return - (lloss_grad + mu * ploss_grad + alpha * norm_loss_grad)

    def loss_split_other2(self, X, y, X1, X2, z, E_z1, E_z2, full_beta, alpha=1, mu=1):
        # full_X = self.fullX
        # K = self.kernel_f(full_X, full_X)
        Kp, Kl1, Kl2 = np.split(self.K, [len(X), len(X) + len(X1)])
        lloss = np.log(1 / (1 + np.exp(-y * self.sigmoid(np.dot(Kp, full_beta)))))
        lloss = lloss.mean()
        ploss = E_z1 * np.log(1 / (1 + np.exp(self.sigmoid(np.dot(Kl1, full_beta)))) * \
                              1 / (1 + np.exp(z * self.sigmoid(np.dot(Kl1, full_beta))))) + \
                E_z2 * np.log(1 / (1 + np.exp(-self.sigmoid(np.dot(Kl1, full_beta)))) * \
                              1 / (1 + np.exp(-z * self.sigmoid(np.dot(Kl1, full_beta)))))
        norm_loss = -np.dot(full_beta.T, full_beta)
        ploss = ploss.mean()
        return - (lloss + mu * ploss + alpha * norm_loss)

    def loss_split(self, X, y, X1, X2, z, E_z1, E_z2, full_beta, alpha=1, mu=1):
        beta, beta1, beta2 = self.split_beta(full_beta)
        A = self.log_loss(X, y, beta, X_prim=X)
        fx1 = self.predict_proba_(X1, beta1, X_prim=X1)
        fx2 = self.predict_proba_(X2, beta2, X_prim=X2)
        pairwise_loss = E_z1 * np.log(1 / (1 + np.exp(fx1)) * 1 / (1 + np.exp(z * fx2))) + \
                        E_z2 * np.log(1 / (1 + np.exp(-fx1)) * 1 / (1 + np.exp(-z * fx2)))
        pairwise_loss = pairwise_loss.mean()
        norm_loss = -np.dot(full_beta.T, full_beta)
        return - (A + mu * pairwise_loss + alpha * norm_loss)

    def loss(self, X, y, X1, X2, z, E_z1, E_z2, full_beta, alpha=1, mu=1):
        # beta, beta1, beta2 = self.split_beta(full_beta)
        A = self.log_loss(X, y, full_beta)
        fx1 = self.predict_proba_(X1, full_beta)
        fx2 = self.predict_proba_(X2, full_beta)
        pairwise_loss = E_z1 * np.log(1 / (1 + np.exp(fx1)) * 1 / (1 + np.exp(z * fx2))) + \
                        E_z2 * np.log(1 / (1 + np.exp(-fx1)) * 1 / (1 + np.exp(-z * fx2)))
        pairwise_loss = pairwise_loss.mean()
        norm_loss = -np.dot(full_beta.T, full_beta)
        return - (A + mu * pairwise_loss + alpha * norm_loss)

    def log_loss(self, X, y, beta, X_prim=None):
        the_loss = np.log(1 / (1 + np.exp(-y * self.predict_proba_(X, beta, X_prim=X_prim))))
        the_loss = np.mean(the_loss)
        return the_loss

    def get_params(self, deep=True):
        return {'alpha': self.alpha,
                'mu': self.mu,
                'percent_pairs': self.percent_pairs,
                'kernel': self.kernel,
                'gamma': self.gamma}

    def set_params(self, **params):
        self.alpha = params.pop('alpha', 1)
        self.mu = params.pop('mu', 1)
        self.percent_pairs = params.pop('percent_pairs', 0.5)
        self.gamma = params.pop('gamma', 'auto')
        self.kernel = params.pop('kernel', 'linear')
        return self


def plot_moons(estimator):
    X, y = make_circles(n_samples=400,
                        noise=0.1)  # make_moons(n_samples=400, noise=0.10, random_state=0)

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.7, random_state=42)
    h = 0.1
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    fig, ax = plt.subplots()

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    estimator.fit(X_train, y_train)
    # score = estimator.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.5, levels=np.linspace(0, 1, 20))


def compare_moons():
    plot_moons(
        # GridSearchCV(1
        LinksClassifier(alpha=2,
                        beta=0.01,
                        kernel='rbf',
                        gamma=3,
                        verbose=True,
                        percent_pairs=0.35,
                        sampling='max_kdist'))

    plot_moons(
        # GridSearchCV(
        LogisticRegression(alpha=0.01,
                           kernel='rbf',
                           gamma=3),
        # {
        #     'alpha': [1e-5, 1e-4, 1e-3],
        #     #'gamma': [0.1, 0.3, 0.5, 0.9]
        # },
        # cv=5,
        # verbose=True,
        # scoring=make_scorer(accuracy_score))
    )
    # plt.show(block=True)
    plot_moons(
        LogisticRegressionPairwise(alpha=0.01,
                                   mu=1,
                                   kernel='rbf',
                                   gamma=3,
                                   verbose=True,
                                   percent_pairs=0.15))


    # LogisticRegression(alpha=1,  kernel='rbf'))
    # plt.show()


if __name__ == '__main__':

    # X, y = make_classification(n_samples=80,
    #                            n_features=20,
    #                            n_informative=15,
    #                            n_classes=3,
    #                            random_state=42)
    # clf = make_pipeline(StandardScaler(),
    #                     LinksClassifier(alpha=2,
    #                                     beta=0.005,
    #                                     kernel='rbf',
    #                                     gamma=0.01,
    #                                     verbose=True,
    #                                     percent_pairs=0.5,
    #                                     sampling='random',
    #                                     init='zeros'))
    #
    # compare = make_pipeline(StandardScaler(),
    #                         SVC(C=1, kernel='rbf', gamma=0.1))
    #
    # kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    #
    # print(cross_val_score(clf,
    #                       X=X, y=y,
    #                       scoring=make_scorer(accuracy_score),
    #                       cv=kf))
    # # print(cross_val_score(compare,
    # #                       X=X, y=y,
    # #                       scoring=make_scorer(accuracy_score),
    # #                       cv=kf))
    # exit()

    compare_moons()
    plt.show()
    exit()


    def load_ds(filename):
        X, y = load_svmlight_file(filename)
        y[y == -1] = 0
        X = X.toarray()
        return X, y


    datasets = OrderedDict([
        ('breast_cancer', load_ds(r'data/breast-cancer_scale.libsvm')),
        ('australian', load_ds(r'data/australian_scale.libsvm')),
        ('hear', load_ds(r'data/heart_scale.libsvm')),
        ('ionosphere', load_ds(r'data/ionosphere_scale.libsvm')),
        ('liver', load_ds(r'data/liver-disorders_scale.libsvm')),
        ('german.numer', load_ds(r'data/german.numer_scale.libsvm')),

    ])

    # for name, (X, y) in datasets.iteritems():
    #     print(name, np.unique(y))
    # exit()

    estimators = [
        ('Logit',
         LogisticRegression(alpha=1),
         {
             'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
         }),
        ('NPLR',
         LogisticRegressionPairwise(alpha=1, mu=1, percent_pairs=0.5),
         {
             'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000],
             'mu': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 1000000]
         }),
        ('SVM(linear)',
         SVC(kernel='linear'),
         {
             'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000],
         })
    ]
    for ds_name, (X, y) in datasets.items()[:1]:
        print('dataset=%s' % ds_name)
        X = StandardScaler().fit_transform(X)


        def plot_gs(ax):
            gs = GridSearchCV(estimator=estimators[1][1], param_grid=estimators[1][2],
                              scoring=make_scorer(accuracy_score),
                              cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                              iid=False, refit=False, n_jobs=-1, verbose=1)
            gs.fit(X, y)
            results = gs.cv_results_


            # plt.plot(results['params'])
            plot_2d_slice(results['mean_test_score'],
                          results['param_alpha'],
                          results['param_mu'], '', ax=ax)
            ax.set_xlabel('mu')
            ax.set_ylabel('alpha')


        fig, ax = plt.subplots(ncols=2)
        plot_gs(ax[0])

        continue

        ax[1].set_xlabel("Training examples")
        ax[1].set_ylabel("Score")
        ax[1].grid()
        fig.suptitle(ds_name)

        for i, (name, clf, grid) in enumerate(estimators):
            print('model=%s' % name)
            gs = GridSearchCV(estimator=clf, param_grid=grid, scoring=make_scorer(accuracy_score),
                              cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
                              iid=False, refit=True, n_jobs=4, verbose=1)

            train_sizes, train_scores, test_scores = learning_curve(
                gs,
                X,
                y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                scoring=make_scorer(accuracy_score),
                verbose=1)

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # scores = cross_val_score(gs, X, y, scoring=make_scorer(accuracy_score),
            #                          cv=StratifiedKFold(), verbose=1)




            # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
            #                  train_scores_mean + train_scores_std, alpha=0.1,
            #                  color="r")
            # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
            #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
            # plt.plot(train_sizes, train_scores_mean, 'o--', color=cm.jet(float(i) / len(estimators)),
            #          label="Training score")
            ax[1].plot(train_sizes, test_scores_mean, 'o-',
                       color=cm.jet(float(i) / len(estimators)),
                       label=name)

            ax[1].legend(loc="best")
    plt.show()
