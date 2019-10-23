import discriminant_analysis as da
import numpy as np
import torch
from sklearn.datasets import make_classification


def main():
    print('LDA')
    X, y = make_classification(n_classes=3, n_features=3, n_informative=3, n_redundant=0,
                               n_clusters_per_class=1, n_samples=300)
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
    print(V)

    # y:
    ys = [torch.mm(w_l, V) for w_l in ws]

    from data_visualizer import DataVisualizer
    dv = DataVisualizer()
    dv.scatter(ws)
    dv.scatter(ys)
    dv.show()


if __name__ == '__main__':
    main()
