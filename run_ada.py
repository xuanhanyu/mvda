import discriminant_analysis as da
import numpy as np
import torch

from torchsl.utils.data_visualizer import DataVisualizer


def normalize(v):
    if isinstance(v, list):
        return [normalize(v_i) for v_i in v]
    elif len(v.shape) == 1:
        return torch.div(v, torch.norm(v, dim=0, p=2))
    return torch.cat([normalize(v_i).unsqueeze(0) for v_i in v], dim=0)


def main():
    print('ADA')

    from sklearn.datasets import make_blobs

    np.random.seed(127)
    X, y = make_blobs(n_features=3, centers=5)
    X += np.ones_like(X) * 10
    y_unique = np.unique(y)
    ws = [torch.tensor(X[np.where(y == y_unique[i])[0]]) for i in range(len(y_unique))]
    ws_norm = normalize(ws)

    # U
    us, u = da.class_means(ws_norm)
    # OW
    Ow = da.within_class_angular_vars(ws_norm, us)
    print(Ow)
    print(da.wcav(normalize(torch.tensor(X, dtype=torch.float)), y))
    print(da.bcav(normalize(torch.tensor(X, dtype=torch.float)), y))
    # OB
    Ob = da.between_class_angular_vars(ws_norm, us)
    # T
    eigen_vals, eigen_vecs = da.eigen((Ow.inverse() @ Ob), argmin=True)
    T = da.projection(eigen_vecs, n_components=2)
    print('[Angular transformation T]', T.shape, sep='\n')

    ys = [torch.cat([(w_i @ T).unsqueeze(0) for w_i in w_l], dim=0) for w_l in ws]
    ys_norm = normalize(ys)
    # ys = [torch.mm(w_l, T) for w_l in ws]
    # ys_norm = [normalize(torch.mm(w_l, T)) for w_l in ws]

    dv = DataVisualizer()
    # dv.scatter(ws)
    # dv.scatter(ys)

    dv.scatter(ws_norm)
    dv.scatter(ys_norm)
    dv.show()


if __name__ == '__main__':
    main()
