from torchsl import *
from dataset.mica_gesture import MultiviewMicaGestureDataset
from data_visualizer import DataVisualizer
from sklearn.manifold.t_sne import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset.utils import join_multiview_datasets
import numpy as np
import torch


def summary(classification_scores):
    single_view_avg = np.diag(classification_scores).mean()
    cross_view_avg = classification_scores - np.eye(classification_scores.shape[0]) * np.diag(classification_scores)
    cross_view_avg = cross_view_avg.sum() / (classification_scores.shape[0] ** 2 - classification_scores.shape[0])
    overall_avg = classification_scores.mean()
    return single_view_avg, cross_view_avg, overall_avg


if __name__ == '__main__':
    visualize = True
    dataset = MultiviewMicaGestureDataset(logic=True)
    dv = DataVisualizer(embed_algo=TSNE(n_components=2), legend=False)

    loo_mv_scores = np.zeros((dataset.n_views, dataset.n_views))
    for loo_id in range(len(dataset)):
        Xs_train, y_train, Xs_test, y_test = dataset[loo_id]
        print(len(y_train), len(y_test))

        # embeds = [PCA(n_components=300, svd_solver='auto') for _ in range(dataset.n_views)]
        # Xs_train = [embeds[_].fit_transform(Xs_train[_], y_train) for _ in range(dataset.n_views)]
        # Xs_test = [embeds[_].transform(Xs_test[_]) for _ in range(dataset.n_views)]

        mvmodel = MvCCDA(n_components='auto', ep_algo='eigen', kernels='linear', lambda_cc=0.001)
        Ys_train = mvmodel.fit_transform(Xs_train, y_train)
        Ys_test = mvmodel.transform(Xs_test)
        print(mvmodel.n_components)
        print('Projected dim', Ys_test[0].shape[1])

        # Classify
        mv_scores = np.zeros((dataset.n_views, dataset.n_views))
        for view_train in range(dataset.n_views):
            for view_test in range(dataset.n_views):
                X_train = Ys_train[view_train]
                X_test = Ys_test[view_test]
                clf = KNeighborsClassifier()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                mv_scores[view_train, view_test] = score
                # print(dataset.views[view_train], dataset.views[view_test], score)
        print(mv_scores)
        print(summary(mv_scores))
        loo_mv_scores += mv_scores

        # np.savetxt("gesturefair_mvda_4096.csv", mv_scores, delimiter=",")
        # exit(0)
        if visualize:
            Xs_all, y_all = join_multiview_datasets([Xs_train, Xs_test], [y_train, y_test])
            Ys_all = join_multiview_datasets([Ys_train, Ys_test])
            # dv.mv_scatter(Xs_all, y_all, title='Original space')
            dv.mv_scatter(Ys_all, y_all, title='Projected space')
            dv.show()
        print()

    print('Overall:')
    loo_mv_scores /= dataset.n_subjects
    print(loo_mv_scores)
    print(summary(loo_mv_scores))
