from torchsl.sl import *
from sklearn.datasets import make_blobs
from sklearn.manifold.t_sne import TSNE
from torchsl.utils import DataVisualizer
from torchsl.grad.constraints import stiefel_restore
from scipy.linalg import fractional_matrix_power
from synthetics import random_dataset
import numpy as np
import torch


def main():
    X, y = make_blobs(n_features=3, centers=3, n_samples=100, random_state=135)
    # y[np.where(y == 2)] = 1

    # from lda_test import X, y

    clf = pcLDA()
    optim = torch.optim.Adam(clf.parameters(), lr=0.01)
    for i in range(100):
        optim.zero_grad()
        loss = clf(X, y)
        loss.backward()
        optim.step()
        print('[{:03d}]'.format(i + 1), 'Loss:', loss.item())

    print(clf.projector.w.t() @ clf.projector.w)
    clf.projector.w.data.copy_(stiefel_restore(clf.projector.w))
    print(clf.projector.w.t() @ clf.projector.w)
    with torch.no_grad():
        Y = clf.transform(X)
    dv = DataVisualizer(embed_algo=TSNE)
    dv.scatter(X, y, title='original')
    dv.scatter(Y, y, title='pcLDA')
    # dv.show(grids=[(1, 2, 0), (1, 2, 1)], title='LDA')
    # exit(0)

    model = LDA(n_components=2, ep_algo='eigen', kernel='none', n_neighbors=4)
    Y = model.fit_transform(X, y)

    # dv.scatter(X, y)
    dv.scatter(Y, y, title='LDA')
    dv.show(grids=[(1, 3, 0), (1, 3, 1), (1, 3, 2)], title='LDA')


if __name__ == '__main__':
    main()
