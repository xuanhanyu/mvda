from .mvdbase import MvDAlgoBase, MvDObjectiveBase
import torch


class MvDA(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis'

    def __init__(self, n_components='auto', ep='eig', reg='auto'):
        super(MvDA, self).__init__(n_components, ep, reg)

    def __Sw__(self):
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

    def __Sb__(self):
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


class MvDAvc(MvDA):
    name = 'Multi-view Discriminant Analysis with View-Consistency'

    def __init__(self, n_components='auto', ep='eig', reg='auto', lambda_vc=0.01):
        super(MvDA, self).__init__(n_components, ep, reg)
        self.lambda_vc = lambda_vc
        self.vco = ViewConsistency(reg=reg)

    def __SwSb__(self):
        self.Sw = self.__Sw__() + self.lambda_vc * self.vco.target()
        self.Sb = self.__Sb__()

    def _post_prepare_(self):
        self.vco.fit_like(self)


class ViewConsistency(MvDObjectiveBase):
    name = 'View-Consistency Objective'

    def __init__(self, reg='auto'):
        super(ViewConsistency, self).__init__()
        self.reg = reg

    def __O__(self):
        Ireg_vc = torch.eye(self.n_samples) * self.reg if not self.reg == 'auto' else None

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    vc_jj = self._Xs[j] @ self._Xs[j].t() @ self._Xs[j] @ self._Xs[j].t()
                    if self.reg == 'auto':
                        vc_jj += torch.eye(self.n_samples) * torch.trace(vc_jj) * 1e-4
                    else:
                        vc_jj += Ireg_vc
                    vc_jr = 2 * self.n_views * vc_jj.inverse() - 2 * vc_jj.inverse()
                else:
                    vc_jr = self._Xs[r] @ self._Xs[r].t() @ self._Xs[j] @ self._Xs[j].t()
                    if self.reg == 'auto':
                        vc_jr += torch.eye(self.n_samples) * torch.trace(vc_jr) * 1e-4
                    else:
                        vc_jr += Ireg_vc
                    vc_jr = -2 * vc_jr.inverse()

                s_jr = self._Xs[j].t() @ vc_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)
