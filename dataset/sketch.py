from scipy.io import loadmat
import torch
import os


class SketchDataset:

    def __init__(self, matfile=os.path.join(os.path.dirname(__file__), 'sketch.mat')):
        data = loadmat(matfile)
        self.Xs_train = torch.from_numpy(data['Xs_train']).float()
        self.Xs_test = torch.from_numpy(data['Xs_test']).float()
        self.y_train = torch.from_numpy(data['y_train']).t().squeeze().long()
        self.y_test = torch.from_numpy(data['y_test']).t().squeeze().long()

    def __call__(self, *args, **kwargs):
        return self.Xs_train, self.y_train, self.Xs_test, self.y_test
