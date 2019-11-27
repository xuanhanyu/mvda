from .bases import BaseSLObjective
from ..utils import affinity
import torch


# ------------------------------------
# LDA
# ------------------------------------
class LDAIntraScatter(BaseSLObjective):

    def __init__(self):
        super(LDAIntraScatter, self).__init__(predicate='minimize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        D = torch.eye(self.n_samples)
        return self._X.t() @ (D - W) @ self._X


class LDAInterScatter(BaseSLObjective):

    def __init__(self):
        super(LDAInterScatter, self).__init__(predicate='maximize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        B = torch.ones(self.n_samples, self.n_samples) / self.n_samples
        return self._X.t() @ (W - B) @ self._X


# ------------------------------------
# LFDA
# ------------------------------------
class LFDAIntraScatter(BaseSLObjective):

    def __init__(self,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1):
        super(LFDAIntraScatter, self).__init__(predicate='minimize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma,
            'row_norm': False
        }

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        A = self.__localize__(W)
        print(A)
        D = A.sum(dim=1).diag()
        return self._X.t() @ (D - A) @ self._X

    def __localize__(self, W):
        return W * affinity(self._X, **self.affinity_params)


class LFDAInterScatter(BaseSLObjective):

    def __init__(self,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1):
        super(LFDAInterScatter, self).__init__(predicate='maximize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma,
            'row_norm': False
        }

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        W = torch.ones(self.n_samples, self.n_samples) / self.n_samples - W
        A = self.__localize__(W)
        D = A.sum(dim=1).diag()
        return self._X.t() @ (D - A) @ self._X

    def __localize__(self, W):
        return W * affinity(self._X, **self.affinity_params)


# ------------------------------------
# LFDA with Locally Linear Embedding
# ------------------------------------
class LFDALLEIntraScatter(BaseSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(LFDALLEIntraScatter, self).__init__(predicate='minimize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma
        }
        self.lambda_lc = lambda_lc

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        B = torch.ones(self.n_samples, self.n_samples) / self.n_samples
        return self._X.t() @ (self.__localize__(W) - B) @ self._X

    def __localize__(self, W):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._X, **self.affinity_params)


class LFDALLEInterScatter(BaseSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(LFDALLEInterScatter, self).__init__(predicate='maximize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma
        }
        self.lambda_lc = lambda_lc

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        B = torch.ones(self.n_samples, self.n_samples) / self.n_samples
        return self._X.t() @ (self.__localize__(W) - B) @ self._X

    def __localize__(self, W):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._X, **self.affinity_params)
