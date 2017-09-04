from scipy.stats import expon, uniform


def get_alpha_distribution(method, n):
    if method == 1:
        return expon(1 / n, n)  # 'exp(n)'
    if method == 2:
        return expon(1 / (10 * n), 10 * n)
    if method == 3:
        return expon(1 / n, n)
    if method == 4:
        return expon(1 / (10 * n), 10 * n)


def get_beta_distribution(method, n):
    if method == 1:
        return expon(1 / n, n)
    if method == 2:
        return expon(1 / (5 * n), 5 * n)
    if method == 3:
        return expon(1 / n, n)
    if method == 4:
        return expon(1 / (10 * n), 10 * n)


def get_delta_distribution(method, n):
    if method == 1:
        return expon(1 / n, n)
    if method == 2:
        return expon(1 / n, n)
    if method == 3:
        return expon(1, 1)
    if method == 4:
        return expon(1 / n, n)


def links_grid_rbf(X, y, fit_kwargs, method=2):
    from grid import links_grid_linear
    grid = links_grid_linear(X, y, fit_kwargs, method=method)
    grid['gamma'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    return grid


def links_grid_rbf_no_labels(X, y, fit_kwargs, method=2):
    from grid import links_grid_linear_no_labels
    grid = links_grid_linear_no_labels(X, y, fit_kwargs, method=method)
    grid['gamma'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    return grid


def links_grid_linear(X, y, fit_kwargs, method=2):
    from grid import get_alpha_distribution, get_beta_distribution, \
        get_delta_distribution
    grid = {}
    if len(y) > 0:
        grid['alpha'] = get_alpha_distribution(method, len(y))

    if 'z' in fit_kwargs and len(fit_kwargs['z']) > 0:
        grid['beta'] = get_beta_distribution(method, len(fit_kwargs['z']))
    if 'Xu' in fit_kwargs and len(fit_kwargs['Xu']) > 0:
        grid['delta'] = get_delta_distribution(method, len(fit_kwargs['Xu']))
    return grid


def links_grid_linear_no_labels(X, y, fit_kwargs, method=2):
    from grid import get_alpha_distribution, get_beta_distribution, \
        get_delta_distribution
    grid = {}
    if len(y) > 0:
        grid['alpha'] = [0]

    if 'z' in fit_kwargs and len(fit_kwargs['z']) > 0:
        grid['beta'] = get_beta_distribution(method, len(fit_kwargs['z']))
    if 'Xu' in fit_kwargs and len(fit_kwargs['Xu']) > 0:
        grid['delta'] = get_delta_distribution(method, len(fit_kwargs['Xu']))
    return grid


def gmm_grid_linear(X, y, fit_kwargs, method=2):
    from grid import get_alpha_distribution, get_beta_distribution, \
        get_delta_distribution
    grid = {}

    if 'z' in fit_kwargs and len(fit_kwargs['z']) > 0:
        grid['positive_prior'] = get_alpha_distribution(method=method, n=len(fit_kwargs['z']))
        grid['negative_prior'] = get_alpha_distribution(method=method, n=len(fit_kwargs['z']))

    grid['delta'] = uniform(1e-15, 1e-2)
    return grid


def svm_grid_rbf(X, y, fit_kwargs, method=2):
    from grid import svm_grid_linear
    grid = svm_grid_linear(X, y, fit_kwargs, method=method)
    grid['gamma'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    return grid


def svm_grid_linear(X, y, fit_kwargs, method=2):
    from grid import get_alpha_distribution
    return {
        'C': get_alpha_distribution(method, len(y)),
    }


def labels_only(fit_kwargs):
    return pop_kwargs(fit_kwargs, to_pop=['X1', 'X2', 'z', 'Xu', 'n_classes'])


def labels_links(fit_kwargs):
    return pop_kwargs(fit_kwargs, to_pop=['Xu'])


def pop_kwargs(kwargs, to_pop):
    for kw in to_pop:
        if kw in kwargs:
            kwargs.pop(kw)
    return kwargs


def prepend(grid_function, prefix):
    return lambda *args, **kwargs: {prefix + key: value
                                    for key, value in (grid_function(*args, **kwargs)).iteritems()}
