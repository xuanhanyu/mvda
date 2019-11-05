from scipy.io import loadmat
import torch
import os


class MultiviewMicaGestureDataset:

    def __init__(self, matfile=os.path.join(os.path.dirname(__file__), 'gesture.mat'), fair=True):
        self.data = loadmat(matfile)
        self.views = [_ for _ in self.data.keys() if not _.startswith('__')]
        self.fair = fair

    def __len__(self):
        return len(self.data[self.views[0]][0])

    def __getitem__(self, item):
        return self.holdout(item)

    def holdout(self, index):
        Xs_train = []
        y_train = []
        Xs_test = []
        y_test = []
        for i, view in enumerate(self.views):
            Xs_train.append(torch.from_numpy(self.data[view][0][index]['X_train'][0][0]).float())
            Xs_test.append(torch.from_numpy(self.data[view][0][index]['X_test'][0][0]).float())
            if i == 0:
                y_train.extend(self.data[view][0][index]['y_train'][0][0][0].tolist())
                y_test.extend(self.data[view][0][index]['y_test'][0][0][0].tolist())
        y_train = torch.tensor(y_train, requires_grad=False).long()
        y_test = torch.tensor(y_test, requires_grad=False).long()
        return Xs_train, y_train, Xs_test, y_test
