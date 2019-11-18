from mvda import *
from data_visualizer import DataVisualizer
from sklearn.manifold.t_sne import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset.sketch import SketchDataset
import numpy as np
import torch


def calc_sim(gallery, test):
    gallery = torch.cat([(_ / torch.norm(_)).unsqueeze(0) for _ in gallery]).squeeze(0)
    test = torch.cat([(_ / torch.norm(_)).unsqueeze(0) for _ in test]).squeeze(0)
    return test @ gallery.t()


def eval_multiview_model(mvmodel, Xs_train, y_train, Xs_test, y_test, return_projected=False):
    Ys_train = mvmodel.fit_transform(Xs_train, y_train)
    Ys_test = mvmodel.transform(Xs_test)
    # Classify
    mv_scores = np.zeros((n_views, n_views))
    for view_train in range(n_views):
        for view_test in range(n_views):
            gallery = Ys_test[view_train]
            test = Ys_test[view_test]
            sim = calc_sim(gallery, test)
            if view_train == view_test:
                sim -= torch.eye(sim.shape[0])
            max_sim, max_sim_index = sim.max(dim=1)
            label_evaluate = y_test[max_sim_index]
            rate = (label_evaluate == y_test).float().sum() / len(y_test)
            mv_scores[view_train, view_test] = rate
    print(mv_scores)
    exit(0)
    return mv_scores


if __name__ == '__main__':
    visualize = True

    dataset = SketchDataset()
    Xs_train, y_train, Xs_test, y_test = dataset()
    n_views = len(Xs_train)
    print(len(y_train), len(y_test))

    mv_scores1 = eval_multiview_model(mvmodel=MvDAvc(n_components=50, ep_algo='ldax', lambda_vc=0.1),
                                      Xs_train=Xs_train, y_train=y_train,
                                      Xs_test=Xs_test, y_test=y_test,
                                      return_projected=True)
    print('Projected space MvDA', mv_scores1, sep='\n', end='\n\n')

    mv_scores2 = eval_multiview_model(mvmodel=MvCSDA(n_components=50, ep_algo='ldax', kernels='linear'),
                                      clf=KNeighborsClassifier(),
                                      Xs_train=Xs_train, y_train=y_train,
                                      Xs_test=Xs_test, y_test=y_test,
                                      return_projected=True)
    print('Projected space MvCSDA', mv_scores2, sep='\n', end='\n\n')

    mv_scores3, Ys_train, Ys_test = eval_multiview_model(
        mvmodel=MvLFDA(n_components=2, ep_algo='eig', kernels='linear', lambda_lc=0.05),
        clf=KNeighborsClassifier(),
        Xs_train=Xs_train, y_train=y_train,
        Xs_test=Xs_test, y_test=y_test,
        return_projected=True)
    print('Projected space MvLFDA', mv_scores3, sep='\n', end='\n\n')

    print(mv_scores2 == mv_scores1)
    print(mv_scores3 >= mv_scores1)
