import discriminant_analysis as da
import numpy as np
import torch
from torchsl.sl import LDA
from sklearn.datasets import make_blobs


def main():
    print('LDA')
    X, y = make_blobs(n_features=3, centers=3, n_samples=100)
    y_unique = np.unique(y)
    ws = [torch.tensor(X[np.where(y == y_unique[i])[0]]) for i in range(len(y_unique))]

    # U
    us, u = da.class_means(ws)
    # SW
    SWs, SW = da.within_class_vars(ws, us)
    # SB
    SBs, SB = da.between_class_vars(ws, us)

    # SlB = da.local_between_class_vars(ws, us)

    # W
    W = SW.inverse() @ SB
    # V:
    eigen_vals, eigen_vecs = da.eigen(W)
    V = da.projection(eigen_vecs, 2)

    # y:
    ys = [torch.mm(w_l, V) for w_l in ws]

    from data_visualizer import DataVisualizer
    dv = DataVisualizer()
    # dv.scatter(ws)
    # dv.scatter(ys)

    model = LDA(n_components=2, ep_algo='ldax', kernel=None)
    Y = model.fit_transform(X, y)
    dv.scatter(X, y)
    dv.scatter(Y, y)
    dv.show()


if __name__ == '__main__':
    main()
