import matlab
import matlab.engine
import torch


class EPSolver:
    def __init__(self, algo='eig', reg='auto'):
        self.reg = reg
        if algo in ['eig', 'eigen']:
            self.solve = self.solve_eig
        elif algo == 'svd':
            self.solve = self.solve_svd
        elif algo == 'matlab':
            print('[EPSolver] Initializing matlab engine for LDAX_SwSb')
            self.engine = matlab.engine.start_matlab()
            self.solve = self.solve_matlab
        self._W = None

    def solve(self, Sw, Sb, argmin=False, *args, **kwargs):
        raise NotImplementedError('Define a solver')

    def solve_eig(self, Sw, Sb, argmin=False):
        if self.reg == 'auto':
            I_reg = torch.eye(Sw.shape[0]) * torch.trace(Sw) * 1e-4
        else:
            I_reg = torch.eye(Sw.shape[0]) * self.reg
        self._W = (Sw + I_reg).inverse() @ Sb
        eigen_vals, eigen_vecs = torch.eig(self._W, eigenvectors=True)
        eigen_pairs = [[eigen_vals[i][0], eigen_vecs.t()[i]] for i in range(eigen_vals.shape[0])]
        eigen_pairs = sorted(eigen_pairs, key=lambda ep: torch.abs(ep[0]).item(), reverse=not argmin)
        eigen_vecs = torch.cat([eigen_pair[1].unsqueeze(0) for eigen_pair in eigen_pairs], dim=0).t()
        return eigen_vecs

    def solve_svd(self, Sw, Sb, argmin=False):
        if self.reg == 'auto':
            I_reg = torch.eye(Sw.shape[0]) * torch.trace(Sw) * 1e-4
        else:
            I_reg = torch.eye(Sw.shape[0]) * self.reg
        self._W = (Sw + I_reg).inverse() @ Sb
        U, S, V = self._W.svd()
        return U[::-1] if argmin else U

    def solve_matlab(self, Sw, Sb, argmin=False):
        if self.engine is None:
            self.engine = matlab.engine.start_matlab()
        self._W = (Sw + torch.eye(Sw.shape[0]) * torch.trace(Sw) * 1e-4).inverse() @ Sb
        ret = self.engine.LDAX_SwSb(matlab.single(Sw.cpu().tolist()), matlab.single(Sb.cpu().tolist()))
        eigen_vecs = torch.tensor(ret._data).view(ret.size).t()
        if argmin:
            eigen_vecs = eigen_vecs[::-1, :]
        return eigen_vecs

    @property
    def meaningful(self):
        return int(torch.matrix_rank(self._W))
