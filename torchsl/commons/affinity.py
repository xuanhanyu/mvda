from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from scipy.sparse import csr_matrix
from scipy.linalg import solve
import numpy as np
import torch


__all__ = ['affinity']


def _barycenter_weights(X, Z, reg=1e-3):
    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B


def _row_norm(af_mat):
    for _ in range(af_mat.shape[0]):
        af_mat[_] = af_mat[_] / torch.sum(af_mat[_])
    return af_mat


def affinity(X,
             algo='lle',
             n_neighbors=5,
             epsilon='auto',
             kernel='rbf', gamma=1, theta=1,
             lle_diag_fill=False,
             row_norm=True,
             n_jobs=-1):
    """
    Compute the affinity matrix.

    :param X:
    :param algo:
    :param n_neighbors:
    :param epsilon:
    :param kernel:
    :param gamma:
    :param lle_diag_fill:
    :param row_norm:
    :param n_jobs:
    :return:
    """
    algo = algo.lower()
    assert algo in ['lle', 'epsilon', 'knn', 'kernel']

    if algo == 'lle':
        # locally linear embedding's reconstruction matrix
        n_neighbors = X.shape[0]
        knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(torch.cat([X, torch.zeros(1, X.shape[1])]))
        X = knn._fit_X
        n_samples = X.shape[0] - 1
        ind = knn.kneighbors(X, return_distance=False)[:-1, 1:]
        data = _barycenter_weights(X[:-1, :], X[ind])
        indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
        af_mat = torch.from_numpy(csr_matrix((data.ravel(), ind.ravel(), indptr),
                                             shape=(n_samples, n_samples)).todense()).float()
        if lle_diag_fill:
            af_mat += torch.eye(n_samples)
        return af_mat

    elif algo == 'knn':
        # k-nearest neighbors
        if n_neighbors < 0:
            n_neighbors = X.shape[0]
        knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(torch.cat([X, torch.zeros(1, X.shape[1])]))
        X = knn._fit_X
        n_samples = X.shape[0] - 1
        ind = knn.kneighbors(X, return_distance=False)
        af_mat = torch.zeros(n_samples, n_samples)
        for i in range(n_samples):
            for j in range(min(n_samples + 1, ind.shape[1])):
                if ind[i, j] < n_samples:
                    if j < n_neighbors:
                        af_mat[i, ind[i, j]] = 1
                    else:
                        break
        # af_mat = torch.from_numpy(csr_matrix((data.ravel(), ind.ravel(), indptr),
        #                                      shape=(n_samples, n_samples)).todense()).float()
        return _row_norm(af_mat) if row_norm else af_mat

    elif algo == 'epsilon':
        # epsilon nearest neighbors
        n_samples = X.shape[0]
        knn = NearestNeighbors(n_samples + 1, n_jobs=n_jobs).fit(torch.cat([X, torch.zeros(1, X.shape[1])]))
        X = knn._fit_X
        dist, ind = knn.kneighbors(X, return_distance=True)
        if isinstance(epsilon, str):
            assert epsilon in ['auto']
            epsilon = np.mean(dist) / 2
        af_mat = torch.zeros(n_samples, n_samples)
        for i in range(n_samples):
            for j in range(n_samples + 1):
                if ind[i, j] < n_samples and dist[i, j] <= epsilon:
                    af_mat[i, ind[i, j]] = 1
        return _row_norm(af_mat) if row_norm else af_mat

    elif algo == 'kernel':
        # heat kernel (rbf or laplacian)
        if isinstance(kernel, str):
            assert kernel in ['rbf', 'laplacian']
            if kernel == 'rbf':
                kernel = rbf_kernel
            elif kernel == 'laplacian':
                kernel = laplacian_kernel
        # else predefined kernel func
        af_mat = torch.from_numpy(kernel(X, gamma=gamma)).float() / theta
        if n_neighbors > 0:
            mask = affinity(X, algo='knn', n_neighbors=n_neighbors, row_norm=False, n_jobs=n_jobs)
            mask -= torch.eye(mask.shape[0])
            print(af_mat[0])
            print(mask[0])
            print()
            af_mat *= mask
        return _row_norm(af_mat) if row_norm else af_mat
