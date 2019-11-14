from scipy.io import loadmat
import torch
import os


class MultiviewMicaGestureDataset:

    def __init__(self, matfile=os.path.join(os.path.dirname(__file__), 'gesture.mat'), logic=True):
        self.data = loadmat(matfile)
        self.views = [_ for _ in self.data.keys() if not _.startswith('__')]
        self.logic = logic
        self.n_views = len(self.views)
        self.n_subjects = len(self)
        if not self.logic:
            self.Xs_fool = []
            for view in self.views:
                X_fool = []
                for test_subject_id in range(self.n_subjects):
                    X_fool.append(torch.from_numpy(self.data[view][0][test_subject_id]['X_test'][0][0]).float())
                self.Xs_fool.append(X_fool)

    def __len__(self):
        return len(self.data[self.views[0]][0])

    def __getitem__(self, item):
        return self.holdout(item)

    def holdout(self, index):
        Xs_train = []
        Xs_test = []
        if self.logic:
            for view in self.views:
                Xs_train.append(torch.from_numpy(self.data[view][0][index]['X_train'][0][0]).float())
                Xs_test.append(torch.from_numpy(self.data[view][0][index]['X_test'][0][0]).float())
        else:
            for i, view in enumerate(self.views):
                Xs_train.append(torch.cat([self.Xs_fool[i][_] for _ in range(self.n_subjects) if _ != index]))
                Xs_test.append(self.Xs_fool[i][index])
        y_train = torch.tensor(self.data[self.views[0]][0][index]['y_train'][0][0][0].tolist(),
                               requires_grad=False).long()
        y_test = torch.tensor(self.data[self.views[0]][0][index]['y_test'][0][0][0].tolist(),
                              requires_grad=False).long()
        return Xs_train, y_train, Xs_test, y_test
