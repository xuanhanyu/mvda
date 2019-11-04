from .mvdbase import MvDObjectiveBase
from .utils import affinity
import torch


# ------------------------------------
# MvDA
# ------------------------------------
class MvDAIntraScatter(MvDObjectiveBase):
    name = 'MvDA Within-class Scatter Objective'

    def __init__(self):
        super(MvDAIntraScatter, self).__init__()

    def __O__(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self.y_unique:
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


class MvDAInterScatter(MvDObjectiveBase):
    name = 'MvDA Between-class Scatter Objective'

    def __init__(self):
        super(MvDAInterScatter, self).__init__()

    def __O__(self):
        n = self.n_views * self.n_samples
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self.y_unique:
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
class ViewConsistency(MvDObjectiveBase):
    name = 'View-Consistency Objective'

    def __init__(self, reg='auto'):
        super(ViewConsistency, self).__init__()
        self.reg = reg
        self.Ireg = torch.eye(self.n_samples) * self.reg if not self.reg == 'auto' else None

    def __regularize__(self, mat):
        if self.reg == 'auto':
            mat += torch.eye(self.n_samples) * torch.trace(mat) * 1e-4
        else:
            mat += self.Ireg
        return mat

    def __O__(self):
        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    vc_jj = self._Xs[j] @ self._Xs[j].t() @ self._Xs[j] @ self._Xs[j].t()
                    vc_jj = self.__regularize__(vc_jj)
                    vc_jr = 2 * self.n_views * vc_jj.inverse() - 2 * vc_jj.inverse()
                else:
                    vc_jr = self._Xs[r] @ self._Xs[r].t() @ self._Xs[j] @ self._Xs[j].t()
                    vc_jr = self.__regularize__(vc_jr)
                    vc_jr = -2 * vc_jr.inverse()

                s_jr = self._Xs[j].t() @ vc_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# Class Separating
# ------------------------------------
class ClassSeparating(MvDObjectiveBase):
    name = 'Class Separating Objective'

    def __init__(self):
        super(ClassSeparating, self).__init__()

    def __O__(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ca in self.y_unique:
            for cb in self.y_unique:
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
class MvLFDAIntraScatter(MvDObjectiveBase):
    name = 'Local Within-class Scatter Objective'

    def __init__(self, affinity_type='kernel', n_neighbors=5, epsilon='auto', kernel='rbf', gamma=1):
        super(MvLFDAIntraScatter, self).__init__()
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': kernel,
            'gamma': gamma
        }

    def __O__(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self.y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0)
        D = torch.eye(self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    W = affinity(self._Xs[j], **self.affinity_params)
                    s_jr = D - W
                else:
                    s_jr = -W
                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


class MvLFDAInterScatter(MvDObjectiveBase):
    name = 'Local Between-class Scatter Objective'

    def __init__(self, affinity_type='kernel', n_neighbors=5, epsilon='auto', kernel='rbf', gamma=1):
        super(MvLFDAInterScatter, self).__init__()
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': kernel,
            'gamma': gamma
        }

    def __O__(self):
        n = self.n_views * self.n_samples
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self.y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0)
        B = torch.ones(self.n_samples, self.n_samples) / n

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    W = affinity(self._Xs[j], **self.affinity_params)
                s_jr = self._Xs[j].t() @ (W - B) @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)
