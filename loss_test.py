from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from sklearn import clone

from sklearn.datasets import make_blobs

from links import LinksClassifier
from new_experiment_runner.cacher import CSVCacher
from new_experiment_runner.runner import Runner
from start_sensitivity import split_dataset
import multiprocess as mp

if __name__ == '__main__':

    alpha_range = [1, 10, 50, 100, 500, 1000]
    beta_range = [1, 10, 50, 100, 500, 1000]
    delta_range = [1, 10, 50, 100, 500, 1000]

    percent_labels_range = [0.05, 0.1, 0.2]
    percent_links_range = [0.05, 0.1, 0.2]
    percent_unlabeled_range = [0.05, 0.1, 0.2]

    percents = list(itertools.izip_longest(percent_labels_range,
                                           percent_links_range,
                                           percent_unlabeled_range,
                                           fillvalue=0))
    params = list(itertools.product(alpha_range, beta_range, delta_range))

    context = OrderedDict()


    def task_generator():
        for std in [1, 2, 3, 4]:
            context['std'] = std
            for trial in xrange(100):
                context['trial'] = trial
                for alpha, beta, delta in params:
                    context['alpha'] = alpha
                    context['beta'] = beta
                    context['delta'] = delta
                    for p_labels, p_links, p_unlab in percents:
                        context['percent_labels'] = p_labels
                        context['percent_links'] = p_links
                        context['percent_unlabeled'] = p_unlab
                        yield OrderedDict(context), {}


    def cross_loss(context, **kwargs):
        from sklearn.datasets import make_blobs
        from start_sensitivity import split_dataset
        from links import LinksClassifier
        from sklearn.base import clone
        X, y = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=context['std'])
        clf1 = LinksClassifier(
            alpha=context['alpha'],
            beta=context['beta'],
            delta=context['delta'],
            sampling='predefined',
            init='normal')
        clf2 = clone(clf1)

        X_1, y_1, X1_1, X2_1, z_1, Xu_1 = split_dataset(X, y,
                                                        context['percent_labels'],
                                                        context['percent_links'],
                                                        context['percent_unlabeled'],
                                                        random_state=None,
                                                        disjoint_labels_and_links=False)
        X_2, y_2, X1_2, X2_2, z_2, Xu_2 = split_dataset(X, y,
                                                        context['percent_labels'],
                                                        context['percent_links'],
                                                        context['percent_unlabeled'],
                                                        random_state=None,
                                                        disjoint_labels_and_links=False)

        clf1.fit(X_1, y_1, X1=X1_1, X2=X2_1, z=z_1, Xu=Xu_1)
        clf2.fit(X_2, y_2, X1=X1_2, X2=X2_2, z=z_2, Xu=Xu_2)

        loss_f1_on_D1 = clf1.calc_loss(X_1, y_1, X1=X1_1, X2=X2_1, z=z_1, Xu=Xu_1)
        loss_f2_on_D1 = clf2.calc_loss(X_1, y_1, X1=X1_1, X2=X2_1, z=z_1, Xu=Xu_1)
        loss_f1_on_D2 = clf1.calc_loss(X_2, y_2, X1=X1_2, X2=X2_2, z=z_2, Xu=Xu_2)
        loss_f2_on_D2 = clf2.calc_loss(X_2, y_2, X1=X1_2, X2=X2_2, z=z_2, Xu=Xu_2)

        return {
            'f1_on_D1_labeled_loss': loss_f1_on_D1[0],
            'f1_on_D1_links_loss': loss_f1_on_D1[1],
            'f1_on_D1_unlabeled_loss': loss_f1_on_D1[2],
            'f1_on_D1_norm_loss': loss_f1_on_D1[3],

            'f2_on_D1_labeled_loss': loss_f2_on_D1[0],
            'f2_on_D1_links_loss': loss_f2_on_D1[1],
            'f2_on_D1_unlabeled_loss': loss_f2_on_D1[2],
            'f2_on_D1_norm_loss': loss_f2_on_D1[3],

            'f2_on_D2_labeled_loss': loss_f2_on_D2[0],
            'f2_on_D2_links_loss': loss_f2_on_D2[1],
            'f2_on_D2_unlabeled_loss': loss_f2_on_D2[2],
            'f2_on_D2_norm_loss': loss_f2_on_D2[3],

            'f1_on_D2_labeled_loss': loss_f1_on_D2[0],
            'f1_on_D2_links_loss': loss_f1_on_D2[1],
            'f1_on_D2_unlabeled_loss': loss_f1_on_D2[2],
            'f1_on_D2_norm_loss': loss_f1_on_D2[3],
        }


    cacher = CSVCacher('data/cross_loss_tests.csv')
    mapper = partial(mp.Pool(processes=7).imap_unordered, chunksize=100)
    runner = Runner(cross_loss, task_generator(), cacher=cacher, mapper=mapper, save_interval=1000)
    runner.run()
    exit()

    X, y = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=1.95, random_state=43)
    fig, ax = plt.subplots(ncols=2)

    clf1 = LinksClassifier(
        alpha=100,
        beta=21,
        delta=0,
        sampling='predefined',
        init='normal')
    clf2 = clone(clf1)

    X_1, y_1, X1_1, X2_1, z_1, Xu_1 = split_dataset(X, y,
                                                    0.1,
                                                    0.05,
                                                    0.1,
                                                    random_state=None,
                                                    disjoint_labels_and_links=False)
    X_2, y_2, X1_2, X2_2, z_2, Xu_2 = split_dataset(X, y,
                                                    0.1,
                                                    0.05,
                                                    0.1,
                                                    random_state=None,
                                                    disjoint_labels_and_links=False)

    clf1.fit(X_1, y_1, X1=X1_1, X2=X2_1, z=z_1, Xu=Xu_1)
    clf2.fit(X_2, y_2, X1=X1_2, X2=X2_2, z=z_2, Xu=Xu_2)

    loss_f1_on_D1 = clf1.calc_loss(X_1, y_1, X1=X1_1, X2=X2_1, z=z_1, Xu=Xu_1)
    loss_f2_on_D1 = clf2.calc_loss(X_1, y_1, X1=X1_1, X2=X2_1, z=z_1, Xu=Xu_1)
    loss_f1_on_D2 = clf1.calc_loss(X_2, y_2, X1=X1_2, X2=X2_2, z=z_2, Xu=Xu_2)
    loss_f2_on_D2 = clf2.calc_loss(X_2, y_2, X1=X1_2, X2=X2_2, z=z_2, Xu=Xu_2)

    print('D1')
    print('model1 on D1: %.3f, %.3f, %.3f, %.3f' % (loss_f1_on_D1[0],
                                                    loss_f1_on_D1[1],
                                                    loss_f1_on_D1[2],
                                                    loss_f1_on_D1[3]))
    print('model2 on D1: %.3f, %.3f, %.3f, %.3f' % (loss_f2_on_D1[0],
                                                    loss_f2_on_D1[1],
                                                    loss_f2_on_D1[2],
                                                    loss_f2_on_D1[3]))
    print('model2 on D2: %.3f, %.3f, %.3f, %.3f' % (loss_f2_on_D2[0],
                                                    loss_f2_on_D2[1],
                                                    loss_f2_on_D2[2],
                                                    loss_f2_on_D2[3]))
    print('model1 on D2: %.3f, %.3f, %.3f, %.3f' % (loss_f1_on_D2[0],
                                                    loss_f1_on_D2[1],
                                                    loss_f1_on_D2[2],
                                                    loss_f1_on_D2[3]))

    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Put the result into a color plot
    Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    Z2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z2 = Z2.reshape(xx.shape)

    ax[0].scatter(X_1[:, 0], X_1[:, 1], c=y_1)
    ax[0].scatter(X1_1[z_1 == 1, 0], X1_1[z_1 == 1, 1], c='green', marker='<')
    ax[0].scatter(X2_1[z_1 == 1, 0], X2_1[z_1 == 1, 1], c='green', marker='>')
    ax[0].scatter(X1_1[z_1 == 0, 0], X1_1[z_1 == 0, 1], c='red', marker='<')
    ax[0].scatter(X2_1[z_1 == 0, 0], X2_1[z_1 == 0, 1], c='red', marker='>')
    for i in xrange(len(z_1)):
        ax[0].plot([X1_1[i, 0], X2_1[i, 0]], [X1_1[i, 1], X2_1[i, 1]],
                   'g-' if z_1[i] else 'r--', alpha=0.5)

    ax[0].scatter(Xu_1[:, 0], Xu_1[:, 1], c='black', marker='o', alpha=0.2)
    ax[1].scatter(X_2[:, 0], X_2[:, 1], c=y_2)
    ax[1].scatter(X1_2[z_2 == 1, 0], X1_2[z_2 == 1, 1], c='green', marker='<')
    ax[1].scatter(X2_2[z_2 == 1, 0], X2_2[z_2 == 1, 1], c='green', marker='>')
    ax[1].scatter(X1_2[z_2 == 0, 0], X1_2[z_2 == 0, 1], c='red', marker='<')
    ax[1].scatter(X2_2[z_2 == 0, 0], X2_2[z_2 == 0, 1], c='red', marker='>')
    ax[1].scatter(Xu_2[:, 0], Xu_2[:, 1], c='black', marker='o', alpha=0.2)
    for i in xrange(len(z_1)):
        ax[1].plot([X1_2[i, 0], X2_2[i, 0]], [X1_2[i, 1], X2_2[i, 1]],
                   'g-' if z_2[i] else 'r--', alpha=0.5)
    ax[0].contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.2)
    ax[1].contourf(xx, yy, Z2, cmap=plt.cm.coolwarm, alpha=0.2)

    plt.show()
