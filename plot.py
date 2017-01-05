import re
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def plot_2d_slice(score_series, values1_series, values2_series, title, ax=None, print_score=True):
    """
    plot 2d slice of grid search results
    :param score_series: mean validation score
    :param values1_series: pd.Series of 1st dimension values
    :param values2_series: pd.Series of 2nd dimension values
    :param title: plot title
    :param ax: matplotlib.Axes to plot onto
    :param print_score: True to print score (already represented by colour on the plot)
    :return: figure
    """
    values1_series = pd.Series(values1_series)
    values2_series = pd.Series(values2_series)
    index = pd.MultiIndex.from_arrays([values1_series, values2_series])
    score_series = pd.Series(score_series, index=index)
    scores = score_series.unstack()

    scores.fillna(0, inplace=True)

    if ax is None:
        fig, ax = plt.subplots()
        # ax = plt

    # scores = score_series.values
    # ax.scatter(values1_series.values, values2_series.values, c=score_series.values, cmap=plt.cm.hot, s=100, marker='s')
    # r = None
    r = ax.imshow(np.round(scores.values, 3), interpolation='nearest', cmap=plt.cm.hot,
                  vmin=scores.values.min(),
                  vmax=scores.values.max())

    ax.set_xticks(np.arange(len(scores.columns)))
    ax.set_xticklabels(scores.columns)
    ax.set_yticks(np.arange(len(scores.index)))
    ax.set_yticklabels(scores.index)
    # ax.set_title(title)


    if print_score:
        for (j, i), v in np.ndenumerate(scores):
            c = (0, 0, 0)
            if v <= 0.5:
                c = (0.8, 0.8, 0.8)
            ax.text(i, j, '%.3g' % np.round(v, 3), va='center', ha='center', color=c)
    return r
