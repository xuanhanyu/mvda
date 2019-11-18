__all__ = ['EPSolver', 'EPAlgo', 'EPImplementation']
try:
    import matlab
    import matlab.engine
except ModuleNotFoundError:
    pass
from ..utils.enum import PredefinedArguments
from ..utils.typing import *
from typing import Union, Tuple
from scipy import linalg
import numpy as np
import torch
import os


class EPAlgo(PredefinedArguments):
    eigen = ['eigen', 'eig']
    svd = ['svd']
    ldax = ['ldax']


class EPImplementation(PredefinedArguments):
    pytorch = ['pytorch', 'torch']
    scipy = ['scipy']
    numpy = ['numpy', 'np']
    matlab = ['matlab']


# Use singleton matlab engine to avoid overhead.
ENGINE: Optional[matlab.engine.matlabengine.MatlabEngine] = None


class EPSolver:

    def __init__(self,
                 algo: Union[EPAlgo, String] = 'eigen',
                 implementation: Union[EPImplementation, String] = 'pytorch',
                 reg: Union[Number, String] = 'auto'):
        assert algo in EPAlgo, 'Undefined algorithm {} for solving Eigen Problem'.format(algo)
        assert implementation in EPImplementation, 'Undefined implementation {}'.format(
            implementation)

        global ENGINE
        self.algo: Optional[EPAlgo] = EPAlgo[algo]
        self.implementation: Optional[EPImplementation] = EPImplementation[implementation]
        self.reg: Number = reg

        if algo == EPAlgo.eigen:
            self.solve = self._solve_eig
        elif algo == EPAlgo.svd:
            self.solve = self._solve_svd
        elif algo == EPAlgo.ldax:
            self.solve = self._solve_ldax
            self.implementation = EPImplementation.matlab

        if implementation == EPImplementation.scipy:
            self.linalg = linalg
        elif implementation == EPImplementation.numpy:
            self.linalg = np.linalg
        elif self.implementation == EPImplementation.matlab:
            if ENGINE is not None:
                self.engine = ENGINE
            else:
                print('[EPSolver] Initializing matlab engine')
                self.engine = matlab.engine.start_matlab()
                self.engine.addpath(os.path.join(os.path.dirname(__file__), 'matlab'))
                ENGINE = self.engine
        self._W: Optional[Tensor] = None

    @property
    def meaningful(self) -> Integer:
        return int(torch.matrix_rank(self._W))

    def solve(self,
              Sw: Tensor,
              Sb: Tensor,
              argmin: Boolean = False,
              *args, **kwargs) -> Tensor:
        raise NotImplementedError('No solver defined')

    def _solve_eig(self,
                   Sw: Tensor,
                   Sb: Tensor,
                   argmin: Boolean = False,
                   *args, **kwargs) -> Tensor:
        Sw, Sb = self.__check_Sw_Sb(Sw, Sb)
        if self.implementation == EPImplementation.pytorch:
            self._W = self.__regularize(Sw).inverse() @ Sb
            evals, evecs = torch.eig(self._W, eigenvectors=True)
            # epairs = [[evals[_][0], evecs.t()[_]] for _ in range(evals.shape[0])]
            # epairs = sorted(epairs, key=lambda ep: torch.abs(ep[0]).item(), reverse=not argmin)
            # evecs = torch.cat([eigen_pair[1].unsqueeze(0) for eigen_pair in epairs], dim=0).t()
            evecs = evecs[:, torch.argsort(evals[:, 0].abs(), descending=not argmin)]
            return evecs
        elif self.implementation == EPImplementation.scipy or self.implementation == EPImplementation.numpy:
            Sw, Sb = self.__numpify(Sw, Sb)
            self._W = self.linalg.inv(self.__regularize(Sw)) @ Sb
            evals, evecs = self.linalg.eig(self._W)
            order = np.argsort(np.abs(evals))[::-1] if not argmin else np.argsort(np.abs(evals))
            evecs = evecs.real[:, order]
            return torch.from_numpy(evecs.astype(np.float32))
        elif self.implementation == EPImplementation.matlab:
            self._W = self.__regularize(Sw).inverse() @ Sb
            W = matlab.single(self._W.cpu().tolist())
            ret = self.engine.sorted_eig(W, not argmin)
            evecs = torch.from_numpy(np.array(ret).real.astype(np.float32)).view(ret.size)
            return evecs

    def _solve_svd(self,
                   Sw: Tensor,
                   Sb: Tensor,
                   argmin: Boolean = False,
                   *args, **kwargs) -> Tensor:
        Sw, Sb = self.__check_Sw_Sb(Sw, Sb)
        if self.implementation == EPImplementation.pytorch:
            self._W = self.__regularize(Sw).inverse() @ Sb
            U = self._W.svd()[0]
            U = self.__revert(U) if argmin else U
            return U
        elif self.implementation == EPImplementation.scipy or self.implementation == EPImplementation.numpy:
            Sw, Sb = self.__numpify(Sw, Sb)
            self._W = self.linalg.inv(self.__regularize(Sw)) @ Sb
            U = self.linalg.svd(self._W)[0]
            U = self.__revert(U) if argmin else U
            return torch.from_numpy(U.astype(np.float32))
        elif self.implementation == EPImplementation.matlab:
            self._W = self.__regularize(Sw).inverse() @ Sb
            W = matlab.single(self._W.cpu().tolist())
            ret = self.engine.single_value_decomposition(W)
            U = torch.tensor(ret._data).view(ret.size).t()
            U = self.__revert(U) if argmin else U
            return U

    def _solve_ldax(self,
                    Sw: Tensor,
                    Sb: Tensor,
                    argmin: Boolean = False,
                    *args, **kwargs) -> Tensor:
        Sw, Sb = self.__check_Sw_Sb(Sw, Sb)
        if self.implementation == EPImplementation.matlab:
            self._W = self.__regularize(Sw).inverse() @ Sb
            ret = self.engine.LDAX_SwSb(matlab.single(Sw.cpu().tolist()), matlab.single(Sb.cpu().tolist()))
            evecs = torch.tensor(ret._data).view(ret.size).t()
            evecs = self.__revert(evecs) if argmin else evecs
            return evecs

    def __check_Sw_Sb(self,
                      Sw: torch.Tensor,
                      Sb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (Sw != 0).any():
            Sw = torch.eye(Sw.shape[0])
        if not (Sb != 0).any():
            Sb = torch.eye(Sb.shape[0])
        return Sw, Sb

    def __numpify(self, *args: torch.Tensor):
        return (arg.numpy() if torch.is_tensor(arg) else arg for arg in args)

    def __matlabfy(self, *args: torch.Tensor):
        return (matlab.single(arg.cpu().tolist()) for arg in args)

    def __regularize(self, Sw: torch.Tensor) -> torch.Tensor:
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

    def __revert(self, mat: Union[torch.Tensor, np.ndarray]):
        if torch.is_tensor(mat):
            return mat.flip([1])
        else:
            return mat[:, ::-1]
