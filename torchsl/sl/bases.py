from ..bases import BaseAlgo
from ..utils import EPSolver, EPAlgo, EPImplementation
from ..utils import Kernelizer
from ..utils import TensorUser, pre_tensorize
from ..utils.typing import *
from abc import ABC, abstractmethod
import torch

# ------------------------------------
# Template for Algorithms
# ------------------------------------
SHARABLE_RESOURCES = {'_X', '_y', '_y_unique', 'ecs', 'n_samples', 'ori_dim', 'dim', 'kernel'}


class BaseSLAlgo(BaseAlgo, TensorUser, ABC):

    def __init__(self,
                 n_components: Union[Integer, String] = 'auto',
                 ep_algo: Union[EPAlgo, String] = 'eig',
                 ep_implementation: Union[EPImplementation, String] = 'pytorch',
                 reg: Union[Number, String] = 'auto',
                 kernel: Optional[Union[Callable, String]] = None,
                 *args, **kwargs):
        super(BaseSLAlgo, self).__init__()
        self.set_sharable_resources(SHARABLE_RESOURCES)
        self.n_components: Integer = n_components
        self.ep_solver: EPSolver = ep_algo if isinstance(ep_algo, EPSolver) else EPSolver(algo=ep_algo,
                                                                                          implementation=ep_implementation,
                                                                                          reg=reg)
        self.reg: Union[Number, String] = reg
        self.kernel: Kernelizer = Kernelizer(kernel)

        # training buffers
        self.Sw: Optional[Tensor] = None
        self.Sb: Optional[Tensor] = None
        self.eig_vecs: Optional[Tensor] = None

    def _fit_(self):
        super()._fit_()
        self.calculate_objectives()
        self.eig_vecs = self.ep_solver.solve(self.Sw, self.Sb)
        if self.n_components == 'auto':
            self.n_components = min(self.ep_solver.meaningful, self.dim)
        elif self.n_components == 'same':
            assert int(self.ori_dim / self.n_views) == self.ori_dim
            self.n_components = self.ori_dim

    @pre_tensorize(positionals=(1, 2), keywords='y_unique')
    def fit(self, X: Tensor,
            y: Union[Tensor, Sequence[Any]],
            y_unique: Optional[Union[Tensor, Sequence[Any]]] = None) -> 'BaseSLAlgo':
        self._prepare_(X, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other: 'BaseAlgo') -> 'BaseSLAlgo':
        assert other.is_fit, '{} is not fitted yet!'.format(other)
        self._prepare_from_(other)
        self._fit_()
        return self

    @pre_tensorize(positionals=(1, 2), keywords='y_unique')
    def fit_transform(self, X: Tensor,
                      y: Union[Tensor, Sequence[Any]],
                      y_unique: Optional[Union[Tensor, Sequence[Any]]] = None) -> Tensor:
        self.fit(X, y, y_unique)
        return self.transform(X)

    def fit_transform_like(self, other: 'BaseAlgo') -> Tensor:
        assert other.is_fit, '{} is not fitted yet!'.format(other)
        _X = getattr(other, '_X')
        self.fit_like(other)
        return self.transform(_X)

    @pre_tensorize(positionals=1)
    def transform(self, X: Tensor) -> Tensor:
        assert self.is_fit
        X = self.kernel.transform(X)
        Y = X @ self.W[:, :self.n_components]
        return Y

    def calculate_objectives(self) -> None:
        self.Sw = self._Sw_()
        self.Sb = self._Sb_()

    @abstractmethod
    def _Sw_(self) -> Tensor:
        pass

    @abstractmethod
    def _Sb_(self) -> Tensor:
        pass

    @property
    def predicates(self) -> Dict:
        predicates = {'maximize': [], 'minimize': []}
        for val in vars(self).values():
            if isinstance(val, BaseSLObjective):
                if val.predicate.lower().startswith('max'):
                    predicates['maximize'].append(str(val))
                elif val.predicate.lower().startswith('min'):
                    predicates['minimize'].append(str(val))
        return predicates

    @property
    def class_vectors(self) -> Tensor:
        return self.ecs

    @property
    def W(self) -> Tensor:
        return self.eig_vecs

    def _prepare_(self,
                  X: Tensor,
                  y: Tensor,
                  y_unique: Optional[Tensor] = None):
        if self.is_prepared:
            return
        self._X = X
        self._y = y
        self._y_unique = torch.unique(self._y) if y_unique is None else y_unique
        self.ecs = torch.stack([torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                                for clazz in self._y_unique])
        self.n_views = self._X.shape[0]
        self.n_samples = self._y.shape[0]
        self.ori_dim = self._X.shape[1]
        self._X = self.kernel.fit_transform(self._X)
        self.dim = self._X.shape[1]
        super()._prepare_()


# ------------------------------------
# Template for Objectives
# ------------------------------------
class BaseSLObjective(BaseAlgo, TensorUser, ABC):

    def __init__(self,
                 predicate: String = 'maximize',
                 kernel: Optional[Union[Callable, String]] = None,
                 *args, **kwargs):
        super(BaseSLObjective, self).__init__()
        self.set_sharable_resources(SHARABLE_RESOURCES)
        self.O = None
        self.predicate = predicate
        self.kernel = Kernelizer(kernel)

    def _fit_(self) -> None:
        self.O = self._O_()
        super()._fit_()

    @pre_tensorize(positionals=(1, 2), keywords='y_unique')
    def fit(self,
            X: Tensor,
            y: Union[Tensor, NumpyArray, Iterable],
            y_unique: Optional[Union[Tensor, NumpyArray, Iterable]] = None) -> 'BaseSLObjective':
        self._prepare_(X, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other: 'BaseAlgo') -> 'BaseSLObjective':
        assert other.is_fit, '{} is not fitted yet!'.format(other)
        self._prepare_from_(other)
        self._fit_()
        return self

    def target(self) -> Optional[Tensor]:
        return self.O

    @abstractmethod
    def _O_(self):
        pass

    def _prepare_(self,
                  X: Tensor,
                  y: Tensor,
                  y_unique: Optional[Tensor] = None) -> None:
        if self.is_prepared:
            return
        self._X = X
        self._y = y
        self._y_unique = torch.unique(self._y) if y_unique is None else y_unique
        self.ecs = torch.stack([torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                                for clazz in self._y_unique])
        self.n_views = self._X.shape[0]
        self.n_samples = self._y.shape[0]
        self.ori_dim = self._X.shape[1]
        self._X = self.kernel.fit_transform(self._X)
        self.dim = self._X.shape[1]
        super()._prepare_()

    def __call__(self, *args, **kwargs):
        return self.O
