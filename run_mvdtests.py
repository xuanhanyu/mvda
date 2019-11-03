from mvda import MvDAvc
import torch
import numpy as np


def group(mv_Ds, y):
    y_unique = np.unique(y)
    mv_Rs = []
    for Ds in mv_Ds:
        mv_Rs.append([Ds[np.where(y == c)[0]] for c in y_unique])
    return mv_Rs


use_kernel = False
if __name__ == '__main__':
    import synthetics
    Xs, y = synthetics.single_blob_dataset()

    # kernelize
    from sklearn.metrics.pairwise import rbf_kernel as kernel
    Ks = [torch.tensor(kernel(Xs[_])).float() if use_kernel else Xs[_] for _ in range(len(Xs))]

    model = MvDAvc(n_components='auto')
    Ys = model.fit_transform(Ks, y)

    # plot
    from data_visualizer import DataVisualizer
    dv = DataVisualizer()
    dv.mv_scatter(Xs, y)
    dv.mv_scatter(Ys, y)
    dv.show()
