from sklearn.datasets import make_blobs, make_gaussian_quantiles
import numpy as np
import torch


def single_blob_dataset():
    np.random.seed(156)
    X_v1, y = make_blobs(n_features=3, centers=3)
    X_v2 = np.array([x + np.random.rand(len(x.shape)) * 3 for x in (X_v1 + np.random.randn(3) * 7)])
    X_v3 = np.array([x + np.random.rand(len(x.shape)) * 3 for x in (X_v1 + np.random.randn(3) * -7)])
    # X_v3, _ = make_blobs(n_features=5, centers=3)
    return [torch.tensor(X_v1).float(), torch.tensor(X_v2).float(), torch.tensor(X_v3).float()], y


def dual_blobs_dataset():
    np.random.seed(138)
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


def gaussian_dataset():
    np.random.seed(154)
    X_v1, y = make_gaussian_quantiles(cov=4.5, n_features=3,
                                      n_classes=3, random_state=156)
    X_v2 = np.array([x + np.random.rand(len(x.shape)) * 2 for x in (X_v1 + np.random.randn(X_v1.shape[1]) * 7)])
    X_v3 = np.array([x + np.random.rand(len(x.shape)) * 1 for x in (X_v1 + np.random.randn(X_v1.shape[1]) * -22)])
    # X_v3, _ = make_blobs(n_features=5, centers=3)
    return [torch.tensor(X_v1).float(), torch.tensor(X_v2).float(), torch.tensor(X_v3).float()], y
