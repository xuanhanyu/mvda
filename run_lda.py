from torchsl.sl import *
from sklearn.datasets import make_blobs
from sklearn.manifold.t_sne import TSNE
from torchsl.utils import DataVisualizer
from synthetics import random_dataset
import numpy as np


def main():
    X, y = make_blobs(n_features=2, centers=3, n_samples=100, random_state=135)
    y[np.where(y == 2)] = 1
    dv = DataVisualizer(embed_algo=TSNE)

    model = LFDA(n_components=1, ep_algo='eigen', kernel=None, n_neighbors=5)
    Y = model.fit_transform(X, y)
    dv.scatter(X, y)
    dv.scatter(Y, y)
    dv.show(grids=[(1, 2, 0), (1, 2, 1)], title='LDA')


if __name__ == '__main__':
    main()
