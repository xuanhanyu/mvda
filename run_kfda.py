import discriminant_analysis as da
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel as kernel_func
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    print('KFDA')
    X, y = make_gaussian_quantiles(n_classes=3, n_features=2, n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    K = kernel_func(X_train)
    y_unique = np.unique(y)
    Xs_train = [X_train[np.where(y_train == y_unique[i])[0]] for i in range(len(y_unique))]
    Ks = [torch.tensor(K[:, np.where(y_train == y_unique[i])[0]], dtype=torch.float) for i in range(len(y_unique))]

    A = da.kfda(Ks)

    K_test = kernel_func(X_test, X_train)
    X_proj = A.numpy().T @ K_test.T
    print(X.shape, X_proj.T.shape)
    Xps_test = [X_proj.T[np.where(y_test == y_unique[i])[0]] for i in range(len(y_unique))]

    from torchsl.utils.data_visualizer import DataVisualizer

    dv = DataVisualizer()
    dv.scatter(Xs_train)
    dv.scatter(Xps_test)
    dv.show()
