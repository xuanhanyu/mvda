from mvda import *
import synthetics
import torch


if __name__ == '__main__':
    precompute_kernel = False
    Xs, y = synthetics.dual_blobs_dataset()

    # kernelize
    from sklearn.metrics.pairwise import rbf_kernel as kernel
    Ks = [torch.tensor(kernel(Xs[_])).float() if precompute_kernel else Xs[_] for _ in range(len(Xs))]

    model = MvLFDA(n_components=2, kernels='none', ep='eig')
    print(model.predicates)
    Ys = model.fit_transform(Xs, y)

    # plot
    from data_visualizer import DataVisualizer
    dv = DataVisualizer()
    dv.mv_scatter(Xs, y)
    dv.mv_scatter(Ys, y)
    dv.show()
