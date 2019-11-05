from mvda import *
from dataset.mica_gesture import MultiviewMicaGestureDataset
from data_visualizer import DataVisualizer
from sklearn.manifold.t_sne import TSNE
from sklearn.neighbors import KNeighborsClassifier
from dataset.utils import join_multiview_datasets
import torch


if __name__ == '__main__':
    dataset = MultiviewMicaGestureDataset()
    dv = DataVisualizer(embed_algo=TSNE(n_components=2), embed_style='global')

    for loo_id in range(len(dataset)):
        Xs_train, y_train, Xs_test, y_test = dataset[loo_id]
        print(len(y_train), len(y_test))

        # Xs_all, y_all = join_multiview_datasets([Xs_train, Xs_test], [y_train, y_test])
        # dv.mv_scatter(Xs_all, y_all)
        # dv.show()

        mvmodel = MvDA(n_components=512, ep='ldax', kernels='linear')
        Ys_train = mvmodel.fit_transform(Xs_train, y_train)
        Ys_test = mvmodel.transform(Xs_test)
        print(mvmodel.n_components)

        Ys_all, y_all = join_multiview_datasets([Ys_train, Ys_test], [y_train, y_test])

        dv.mv_scatter(Ys_all, y_all)
        dv.show()
