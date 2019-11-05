try:
    import matlab
    import matlab.engine
except ModuleNotFoundError:
    pass
from scipy import linalg
import numpy as np
import torch
import os


# Use singleton matlab engine to avoid overhead.
ENGINE = None


class EPSolver:

    def __init__(self, algo='eig', reg='auto', implementation='matlab'):
        assert implementation in ['pytorch', 'scipy', 'numpy', 'matlab']
        global ENGINE
        self.implementation = implementation
        self.reg = reg
        if algo in ['eig', 'eigen']:
            self.solve = self.solve_eig
        elif algo == 'svd':
            self.solve = self.solve_svd
        elif algo == 'ldax':
            # raise ChildProcessError('LDAX_SwSb.p solver is very unreliable!'
            #                         'I don\'t know why they have to hide their source code.'
            #                         'I can not debug their faults, consider using others.')
            print('[EPSolver] Initializing matlab engine')
            self.solve = self.solve_ldax
            self.implementation = 'matlab'

        if self.implementation in ['scipy', 'numpy']:
            if self.implementation == 'scipy':
                self.linalg = linalg
            else:
                self.linalg = np.linalg
        if self.implementation == 'matlab':
            if ENGINE is not None:
                self.engine = ENGINE
            else:
                self.engine = matlab.engine.start_matlab()
                self.engine.addpath(os.path.join(os.path.dirname(__file__), 'matlab'))
                ENGINE = self.engine

        self._W = None

    def solve(self, Sw, Sb, argmin=False, *args, **kwargs):
        raise NotImplementedError('Define a solver')

    def solve_eig(self, Sw, Sb, argmin=False):
        Sw, Sb = self.__check__(Sw, Sb)
        if self.implementation == 'pytorch':
            self._W = self.__regularize__(Sw).inverse() @ Sb
            evals, evecs = torch.eig(self._W, eigenvectors=True)
            # epairs = [[evals[_][0], evecs.t()[_]] for _ in range(evals.shape[0])]
            # epairs = sorted(epairs, key=lambda ep: torch.abs(ep[0]).item(), reverse=not argmin)
            # evecs = torch.cat([eigen_pair[1].unsqueeze(0) for eigen_pair in epairs], dim=0).t()
            evecs = evecs[:, torch.argsort(evals[:, 0].t().squeeze(), descending=not argmin)]
            return evecs
        elif self.implementation in ['scipy', 'numpy']:
            Sw, Sb = self.__numpify__(Sw, Sb)
            self._W = self.linalg.inv(self.__regularize__(Sw)) @ Sb
            evals, evecs = self.linalg.eig(self._W)
            order = np.argsort(evals)[::-1] if not argmin else np.argsort(evals)
            evecs = evecs.real[:, order]
            return torch.from_numpy(evecs.astype(np.float32))
        elif self.implementation == 'matlab':
            self._W = self.__regularize__(Sw).inverse() @ Sb
            W = matlab.single(self._W.cpu().tolist())
            ret = self.engine.sorted_eig(W, not argmin)
            evecs = torch.from_numpy(np.array(ret).real.astype(np.float32)).view(ret.size)
            return evecs

    def solve_svd(self, Sw, Sb, argmin=False):
        Sw, Sb = self.__check__(Sw, Sb)
        if self.implementation == 'pytorch':
            self._W = self.__regularize__(Sw).inverse() @ Sb
            U = self._W.svd()[0]
            U = self.__revert__(U) if argmin else U
            return U
        elif self.implementation in ['scipy', 'numpy']:
            Sw, Sb = self.__numpify__(Sw, Sb)
            self._W = self.linalg.inv(self.__regularize__(Sw)) @ Sb
            U = self.linalg.svd(self._W)[0]
            U = self.__revert__(U) if argmin else U
            return torch.from_numpy(U.astype(np.float32))
        elif self.implementation == 'matlab':
            self._W = self.__regularize__(Sw).inverse() @ Sb
            W = matlab.single(self._W.cpu().tolist())
            ret = self.engine.single_value_decomposition(W)
            U = torch.tensor(ret._data).view(ret.size).t()
            U = self.__revert__(U) if argmin else U
            return U

    def solve_ldax(self, Sw, Sb, argmin=False):
        Sw, Sb = self.__check__(Sw, Sb)
        if self.implementation == 'matlab':
            self._W = self.__regularize__(Sw).inverse() @ Sb
            ret = self.engine.LDAX_SwSb(matlab.single(Sw.cpu().tolist()), matlab.single(Sb.cpu().tolist()))
            evecs = torch.tensor(ret._data).view(ret.size).t()
            evecs = self.__revert__(evecs) if argmin else evecs
            return evecs

    @property
    def meaningful(self):
        return int(torch.matrix_rank(self._W))

    def __check__(self, Sw, Sb):
        if not (Sw != 0).any():
            Sw = torch.eye(Sw.shape[0])
        if not (Sb != 0).any():
            Sb = torch.eye(Sb.shape[0])
        return Sw, Sb

    def __numpify__(self, *args):
        return (arg.numpy() if torch.is_tensor(arg) else arg for arg in args)

    def __matlabfy__(self, *args):
        return (matlab.single(arg.cpu().tolist()) for arg in args)

    def __regularize__(self, Sw):
        if torch.is_tensor(Sw):
            if self.reg == 'auto':
                I_reg = torch.eye(Sw.shape[0]) * torch.trace(Sw) * 1e-4
            else:
                I_reg = torch.eye(Sw.shape[0]) * self.reg
        else:
            if self.reg == 'auto':
                I_reg = np.eye(Sw.shape[0]) * np.trace(Sw) * 1e-4
            else:
                I_reg = np.eye(Sw.shape[0]) * self.reg
        return Sw + I_reg

    def __revert__(self, mat):
        if torch.is_tensor(mat):
            return mat.flip([1])
        else:
            return mat[:, ::-1]

    # def __del__(self):
    #     if hasattr(self, 'engine'):
    #         try:
    #             self.engine.quit()
    #         except:
    #             pass
