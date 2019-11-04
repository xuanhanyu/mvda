from mvda import *
import torch


use_kernel = False
if __name__ == '__main__':
    import synthetics
    Xs, y = synthetics.single_blob_dataset()

    # kernelize
    from sklearn.metrics.pairwise import rbf_kernel as kernel
    Ks = [torch.tensor(kernel(Xs[_])).float() if use_kernel else Xs[_] for _ in range(len(Xs))]

    model = MvDAplusCS(n_components=2)
    Ys = model.fit_transform(Ks, y)

    # plot
    from data_visualizer import DataVisualizer
    dv = DataVisualizer()
    dv.mv_scatter(Xs, y)
    dv.mv_scatter(Ys, y)
    dv.show()
