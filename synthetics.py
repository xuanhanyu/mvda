from sklearn.datasets import make_blobs, make_gaussian_quantiles
from scipy.stats import ortho_group
import numpy as np
import torch


def rvs(dim=3, seed=None):
    np.random.seed(seed)
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def random_dataset(n_classes=3, n_views=3, n_features=3, n_samples='auto', rotate=True, shuffle=True, seed=None):
    np.random.seed(seed)
    n_samples = n_classes * 20 if n_samples == 'auto' else n_samples
    X_ori, y = make_blobs(n_features=n_features, centers=n_classes, n_samples=n_samples)
    Xs = [X_ori]
    for i in range(n_views - 1):
        X_new_view = X_ori + np.random.randn(n_features) * np.random.randint(1, 15)  # Translate
        X_new_view = np.array([x + np.random.rand(len(x.shape)) * np.random.randint(1, 5)
                               for x in X_new_view])
        if rotate:
            X_new_view = X_new_view @ rvs(n_features, seed)
        Xs.append(X_new_view)
    if shuffle:
        indexes = np.random.permutation(np.arange(len(y)))
        y = y[indexes]
        for _ in range(n_views):
            Xs[_] = Xs[_][indexes, :]
    if n_views > 1:
        return torch.stack([torch.tensor(X).float() for X in Xs]), y
    return torch.tensor(Xs).squeeze(0).float(), y


def single_blob_dataset(n_classes=3, n_views=3, n_features=3, n_samples='auto', rotate=True, shuffle=True, seed=156):
    np.random.seed(seed)
    n_samples = n_classes * 20 if n_samples == 'auto' else n_samples
    X_ori, y = make_blobs(n_features=n_features, centers=n_classes, n_samples=n_samples)
    Xs = [X_ori]
    for i in range(n_views - 1):
        X_new_view = X_ori + np.random.randn(n_features) * 7
        X_new_view = np.array([x + np.random.rand(len(x.shape)) * 3 for x in X_new_view])
        if rotate:
            X_new_view = X_new_view @ rvs(n_features, seed)
        Xs.append(X_new_view)
    if shuffle:
        indexes = np.random.permutation(np.arange(len(y)))
        y = y[indexes]
        for _ in range(n_views):
            Xs[_] = Xs[_][indexes, :]
    return [torch.tensor(X).float() for X in Xs], y


def dual_blobs_dataset(seed=138):
    np.random.seed(seed)
    X_v1, y = make_blobs(n_features=3, centers=3)
    X_v2 = np.array([x + np.random.rand(len(x.shape)) * 3 for x in (X_v1 + np.random.randn(3) * 7)])
    X_v3 = np.array([x + np.random.rand(len(x.shape)) * 3 for x in (X_v1 + np.random.randn(3) * -7)])
    # X_v3, _ = make_blobs(n_features=5, centers=3)
    for _ in range(y.shape[0]):
        if y[_] == 0:
            y[_] = 1
        else:
            y[_] = 0
    return [torch.tensor(X_v1).float(), torch.tensor(X_v2).float(), torch.tensor(X_v3).float()], y


def gaussian_dataset(n_classes=3, n_views=3, n_features=3, n_samples='auto', rotate=True, shuffle=True, seed=154):
    np.random.seed(seed)
    n_samples = n_classes * 20 if n_samples == 'auto' else n_samples
    X_ori, y = make_gaussian_quantiles(cov=4.5, n_features=3, n_samples=n_samples,
                                       n_classes=n_classes, random_state=156)
    Xs = [X_ori]
    for i in range(n_views - 1):
        X_new_view = X_ori + np.random.randn(n_features) * np.random.randint(7, 30)
        X_new_view = np.array([x + np.random.rand(len(x.shape)) * np.random.randint(1, 3) for x in X_new_view])
        if rotate:
            X_new_view = X_new_view @ rvs(n_features, seed)
        Xs.append(X_new_view)
    if shuffle:
        indexes = np.random.permutation(np.arange(len(y)))
        y = y[indexes]
        for _ in range(n_views):
            Xs[_] = Xs[_][indexes, :]
    return [torch.tensor(X).float() for X in Xs], y
