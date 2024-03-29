from .bases import BaseMvSLObjective
from ..commons import affinity
import torch


# ------------------------------------
# MvDA
# ------------------------------------
class MvDAIntraScatter(BaseMvSLObjective):

    def __init__(self):
        super(MvDAIntraScatter, self).__init__(predicate='minimize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * self.n_views)
        D = torch.eye(self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    s_jr = D - W
                else:
                    s_jr = -W

                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


class MvDAInterScatter(BaseMvSLObjective):

    def __init__(self):
        super(MvDAInterScatter, self).__init__(predicate='maximize')

    def _O_(self):
        n = self.n_views * self.n_samples
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * len(self._Xs))
        B = torch.ones(self.n_samples, self.n_samples) / n

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                s_jr = self._Xs[j].t() @ (W - B) @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# MvDA-vc
# ------------------------------------
class ViewConsistency(BaseMvSLObjective):

    def __init__(self, reg='auto'):
        super(ViewConsistency, self).__init__(predicate='minimize')
        self.reg = reg
        if not self.reg == 'auto':
            self.Ireg = torch.eye(self.n_samples) * self.reg

    def __regularize__(self, mat):
        if self.reg == 'auto':
            mat += torch.eye(self.n_samples) * torch.trace(mat) * 1e-4
        else:
            mat += self.Ireg
        return mat

    def _O_(self):
        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    k_j = self._Xs[j] @ self._Xs[j].t() if not self.kernels.statuses[j] else self._Xs[j]

                    # vc_jj = self._Xs[j] @ self._Xs[j].t() @ self._Xs[j] @ self._Xs[j].t()
                    vc_jj = self.__regularize__(k_j @ k_j)
                    vc_jr = 2 * self.n_views * vc_jj.inverse() - 2 * vc_jj.inverse()
                else:
                    k_j = self._Xs[j] @ self._Xs[j].t() if not self.kernels.statuses[j] else self._Xs[j]
                    k_r = self._Xs[r] @ self._Xs[r].t() if not self.kernels.statuses[r] else self._Xs[r]

                    # vc_jr = self._Xs[r] @ self._Xs[r].t() @ self._Xs[j] @ self._Xs[j].t()
                    vc_jr = self.__regularize__(k_r @ k_j)
                    vc_jr = -2 * vc_jr.inverse()

                s_jr = self._Xs[j].t() @ vc_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# Class Separating
# ------------------------------------
class ClassSeparating(BaseMvSLObjective):

    def __init__(self):
        super(ClassSeparating, self).__init__(predicate='maximize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ca in self._y_unique:
            for cb in self._y_unique:
                W += torch.sum(self.ecs[cb]) / torch.sum(self.ecs[ca]) * self.ecs[ca].unsqueeze(0).t() @ self.ecs[ca].unsqueeze(0)
        D = torch.ones(self.n_samples, self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                vc_jr = 2 * W - 2 * D

                s_jr = self._Xs[j].t() @ vc_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# MvLFDA
# ------------------------------------
class MvLFDAIntraScatter(BaseMvSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(MvLFDAIntraScatter, self).__init__(predicate='minimize')
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
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * self.n_views)
        D = torch.eye(self.n_samples)
        # W = torch.zeros(self.n_samples, self.n_samples)
        # for ci in self._y_unique:
        #     W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0)
        # D = torch.eye(self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                W_af = self.__localize__(W, j)
                if j == r:
                    s_jr = D - W_af
                else:
                    s_jr = -W
                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)

    def __localize__(self, W, j):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._Xs[j], **self.affinity_params)


class MvLFDAInterScatter(BaseMvSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(MvLFDAInterScatter, self).__init__(predicate='maximize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma
        }
        self.lambda_lc = lambda_lc

    def _O_(self):
        n = self.n_views * self.n_samples
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * len(self._Xs))
        B = torch.ones(self.n_samples, self.n_samples) / n
        # n = self.n_views * self.n_samples
        # W = torch.zeros(self.n_samples, self.n_samples)
        # for ci in self._y_unique:
        #     W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0)
        # B = torch.ones(self.n_samples, self.n_samples) / n

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                W_af = self.__localize__(W, j)
                s_jr = self._Xs[j].t() @ (W_af - B) @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)

    def __localize__(self, W, j):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._Xs[j], **self.affinity_params)


# ------------------------------------
# MvCCDA
# ------------------------------------
class CommonComponent(BaseMvSLObjective):

    def __init__(self):
        super(CommonComponent, self).__init__(predicate='minimize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * self.n_views)

        I = torch.eye(self.n_samples)
        D = torch.ones(self.n_samples, self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    s_jr = 2 / self.n_views * (I - D)
                else:
                    s_jr = 2 / self.n_views * D
                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


class DifferingClass(BaseMvSLObjective):

    def __init__(self):
        super(DifferingClass, self).__init__(predicate='maximize')

    def _O_(self):
        I = torch.eye(self.n_samples) * self.n_views
        E = torch.zeros(self.n_samples, self.n_samples)
        for ca in self._y_unique:
            for cb in self._y_unique:
                if ca != cb:
                    E += self.ecs[ca].unsqueeze(0).t() @ self.ecs[cb].unsqueeze(0)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    s_jr = 2 * (I - E)
                else:
                    s_jr = 2 * -E
                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# Regularizer
# ------------------------------------
class Regularization(BaseMvSLObjective):

    def __init__(self):
        super(Regularization, self).__init__(predicate='minimize')

    def _O_(self):
        return torch.eye(sum(self.dims))
