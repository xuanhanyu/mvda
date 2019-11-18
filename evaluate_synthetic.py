from mvda import *
from dataset.mica_gesture import MultiviewMicaGestureDataset
from data_visualizer import DataVisualizer
from sklearn.manifold.t_sne import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset.utils import multiview_train_test_split, join_multiview_datasets
import numpy as np
import synthetics
import torch


def eval_multiview_model(mvmodel, clf, Xs_train, y_train, Xs_test, y_test, return_projected=False):
    Ys_train = mvmodel.fit_transform(Xs_train, y_train)
    Ys_test = mvmodel.transform(Xs_test)
    # Classify
    mv_scores = np.zeros((n_views, n_views))
    for view_train in range(n_views):
        for view_test in range(n_views):
            X_train = Ys_train[view_train]
            X_test = Ys_test[view_test]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            mv_scores[view_train, view_test] = score
    if return_projected:
        return mv_scores, Ys_train, Ys_test
    return mv_scores


if __name__ == '__main__':
    visualize = True
    Xs, y = synthetics.single_blob_dataset(n_classes=5, n_views=3, n_features=3, seed=107)  # 107
    dv = DataVisualizer(embed_algo=TSNE(n_components=3), embed_style='global', legend=False)
    n_views = len(Xs)
    Xs_train, y_train, Xs_test, y_test = multiview_train_test_split(Xs, y)
    print(len(y_train), len(y_test))

    Xs_all, y_all = join_multiview_datasets([Xs_train, Xs_test], [y_train, y_test])
    dv.mv_scatter(Xs_all, y_all, title='Original space')

    mv_scores1, Ys_train, Ys_test = eval_multiview_model(mvmodel=MvDA(n_components=2, ep_algo='svd', ep_implementation='matlab', kernels='linear'),
                                                         clf=KNeighborsClassifier(),
                                                         Xs_train=Xs_train, y_train=y_train,
                                                         Xs_test=Xs_test, y_test=y_test,
                                                         return_projected=True)
    print('Projected space MvDA', mv_scores1, sep='\n', end='\n\n')
    Ys_all = join_multiview_datasets([Ys_train, Ys_test])
    dv.mv_scatter(Ys_all, y_all, title='Projected space MvDA')

    # mv_scores2, Ys_train, Ys_test = eval_multiview_model(mvmodel=MvCSDA(n_components=2, ep='eig', kernels='linear'),
    #                                                      clf=KNeighborsClassifier(),
    #                                                      Xs_train=Xs_train, y_train=y_train,
    #                                                      Xs_test=Xs_test, y_test=y_test,
    #                                                      return_projected=True)
    # print('Projected space MvCSDA', mv_scores2, sep='\n', end='\n\n')
    # Ys_all = join_multiview_datasets([Ys_train, Ys_test])
    # dv.mv_scatter(Ys_all, y_all, title='Projected space MvCSDA')
    #
    # mv_scores3, Ys_train, Ys_test = eval_multiview_model(mvmodel=MvLFDA(n_components=2, ep='eig', kernels='linear', lambda_lc=0.05),
    #                                                      clf=KNeighborsClassifier(),
    #                                                      Xs_train=Xs_train, y_train=y_train,
    #                                                      Xs_test=Xs_test, y_test=y_test,
    #                                                      return_projected=True)
    # print('Projected space MvLFDA', mv_scores3, sep='\n', end='\n\n')
    # Ys_all = join_multiview_datasets([Ys_train, Ys_test])
    # dv.mv_scatter(Ys_all, y_all, title='Projected space MvLFDA')

    # np.savetxt("gesturefair_mvda_4096.csv", mv_scores, delimiter=",")
    # print(mv_scores2 == mv_scores1)
    # print(mv_scores3 >= mv_scores1)
    dv.show()
