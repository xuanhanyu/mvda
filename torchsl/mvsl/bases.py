from ..bases import BaseAlgo, MetaEOBasedAlgo, MetaGradientBasedAlgo
from ..commons import EPSolver, EPAlgo, EPImplementation
from ..commons import MvKernelizer
from ..utils import TensorUser, pre_tensorize
from ..utils.typing import *
from abc import ABC, abstractmethod
import torch

# ------------------------------------
# Template for Algorithms
# ------------------------------------
SHARABLE_RESOURCES = {'_Xs', '_y', '_y_unique', 'ecs', 'n_views', 'n_classes', 'n_samples', 'ori_dims', 'dims', 'kernels'}


class AbstractMvSLAlgo(BaseAlgo, TensorUser):

    def __init__(self, reg='auto'):
        BaseAlgo.__init__(self, sharable_resources=SHARABLE_RESOURCES)
        TensorUser.__init__(self, reg=reg)

    def _prepare_(self,
                  Xs: Tensor,
                  y: Tensor,
                  y_unique: Optional[Tensor] = None):
        # if self.is_prepared and not self.__should_reprepare:
        #     return
        self._Xs = Xs
        self._y = y
        self._y_unique = torch.unique(self._y) if y_unique is None else y_unique
        self.ecs = torch.stack([torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                                for clazz in self._y_unique])
        self.n_views = self._Xs.shape[0]
        self.n_classes = len(self._y_unique)
        self.n_samples = self._y.shape[0]
        self.ori_dims = [X.shape[1] for X in self._Xs]
        self.dims = [X.shape[1] for X in self._Xs]


class EOBasedMvSLAlgo(AbstractMvSLAlgo, metaclass=MetaEOBasedAlgo):

    def __init__(self,
                 n_components: Union[Integer, String] = 'auto',
                 ep_algo: Union[EPAlgo, String] = 'eig',
                 ep_implementation: Union[EPImplementation, String] = 'pytorch',
                 reg: Union[Number, String] = 'auto',
                 kernels: Optional[Union[Callable, String, Iterable[Callable], Iterable[String]]] = None,
                 *args, **kwargs):
        AbstractMvSLAlgo.__init__(self, reg=reg)
        self.n_components: Integer = n_components
        self.ep_solver: EPSolver = ep_algo if isinstance(ep_algo, EPSolver) else EPSolver(algo=ep_algo,
                                                                                          implementation=ep_implementation,
                                                                                          reg=reg)
        self.kernels: MvKernelizer = MvKernelizer(kernels)

        # training buffers
        self.Sw: Optional[Tensor] = None
        self.Sb: Optional[Tensor] = None
        self.eig_vecs: Optional[Tensor] = None
        self.ws: Optional[Tensor] = None

    def _fit_(self):
        super()._fit_()
        self.calculate_objectives()
        self.eig_vecs = self.ep_solver.solve(self.Sw, self.Sb)
        if self.n_components == 'auto':
            self.n_components = min(self.ep_solver.meaningful, min(self.dims))
        elif self.n_components == 'same':
            assert int(sum(self.ori_dims) / self.n_views) == self.ori_dims[0]
            self.n_components = self.ori_dims[0]
        self.ws = torch.stack(
            [self.eig_vecs[sum(self.dims[:i]):sum(self.dims[:i + 1]), :] for i in range(len(self.dims))])

    def fit(self, Xs: Union[Tensor, Sequence[Tensor]],
            y: Union[Tensor, Sequence[Any]],
            y_unique: Optional[Union[Tensor, Sequence[Any]]] = None) -> 'EOBasedMvSLAlgo':
        self._prepare_(Xs, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other: 'BaseAlgo') -> 'EOBasedMvSLAlgo':
        assert other.is_fit, '{} is not fitted yet!'.format(other)
        self._prepare_from_(other)
        self._fit_()
        return self

    def fit_transform(self, Xs: Union[Tensor, Sequence[Tensor]],
                      y: Union[Tensor, Sequence[Any]],
                      y_unique: Optional[Union[Tensor, Sequence[Any]]] = None) -> Tensor:
        self.fit(Xs, y, y_unique)
        return self.transform(Xs)

    def fit_transform_like(self, other: 'BaseAlgo') -> Tensor:
        assert other.is_fit, '{} is not fitted yet!'.format(other)
        _Xs = getattr(other, '_Xs')
        self.fit_like(other)
        return self.transform(_Xs)

    def transform(self, Xs: Union[Tensor, Sequence[Tensor]]) -> Tensor:
        assert self.is_fit
        Xs = self.kernels.transform(Xs)
        Ys = torch.stack([Xs[_] @ self.ws[_][:, :self.n_components] for _ in range(Xs.shape[0])])
        return Ys

    def calculate_objectives(self) -> None:
        self.Sw = self._Sw_()
        self.Sb = self._Sb_()

    def _Sw_(self) -> Tensor:
        pass

    def _Sb_(self) -> Tensor:
        pass

    @property
    def predicates(self) -> Dict:
        predicates = {'maximize': [], 'minimize': []}
        for val in vars(self).values():
            if isinstance(val, BaseMvSLObjective):
                if val.predicate.lower().startswith('max'):
                    predicates['maximize'].append(str(val))
                elif val.predicate.lower().startswith('min'):
                    predicates['minimize'].append(str(val))
        return predicates

    @property
    def class_vectors(self) -> Tensor:
        return self.ecs

    @property
    def projections(self) -> Tensor:
        return self.ws

    @property
    def W(self) -> Tensor:
        return torch.cat([w for w in self.ws])

    def _prepare_(self,
                  Xs: Union[Tensor],
                  y: Tensor,
                  y_unique: Optional[Tensor] = None):
        super()._prepare_(Xs, y, y_unique)
        self._Xs = self.kernels.fit_transform(self._Xs)
        self.dims = [X.shape[1] for X in self._Xs]
        self._post_prepare_()


class GradientBasedMvSLAlgo(AbstractMvSLAlgo, torch.nn.Module, metaclass=MetaGradientBasedAlgo):

    def __init__(self, reg='auto'):
        AbstractMvSLAlgo.__init__(self, reg=reg)
        torch.nn.Module.__init__(self)

    def forward(self,
                Xs: Tensor,
                y: Tensor,
                y_unique: Optional[Tensor] = None) -> Tensor:
        pass

    def transform(self, Xs: Tensor) -> Tensor:
        pass

    def _prepare_(self,
                  Xs: Tensor,
                  y: Tensor,
                  y_unique: Optional[Tensor] = None):
        super()._prepare_(Xs, y, y_unique)
        self._post_prepare_()


# ------------------------------------
# Template for Objectives
# ------------------------------------
class BaseMvSLObjective(AbstractMvSLAlgo):

    def __init__(self,
                 predicate: String = 'maximize',
                 kernels: Optional[Union[Callable, String, Iterable[Callable], Iterable[String]]] = None,
                 reg: Union[Number, String] = 'auto',
                 *args, **kwargs):
        AbstractMvSLAlgo.__init__(self, reg=reg)
        self.O = None
        self.predicate = predicate
        self.kernels = MvKernelizer(kernels)

    def _fit_(self) -> None:
        self.O = self._O_()
        super()._fit_()

    @pre_tensorize(positionals=1, dtype=torch.float)
    @pre_tensorize(positionals=(2, 3), keywords='y_unique', dtype=torch.long)
    def fit(self,
            Xs: Union[Tensor, Iterable[Tensor]],
            y: Union[Tensor, NumpyArray, Iterable],
            y_unique: Optional[Union[Tensor, NumpyArray, Iterable]] = None) -> 'BaseMvSLObjective':
        self._prepare_(Xs, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other: 'BaseAlgo') -> 'BaseMvSLObjective':
        assert other.is_fit, '{} is not fitted yet!'.format(other)
        self._prepare_from_(other)
        self._fit_()
        return self

    def target(self) -> Optional[Tensor]:
        return self.O

    def _O_(self):
        pass

    def _prepare_(self,
                  Xs: Tensor,
                  y: Tensor,
                  y_unique: Optional[Tensor] = None) -> None:
        super()._prepare_(Xs, y, y_unique)
        # if self.is_prepared and not self.__should_reprepare:
        #     return
        # self._Xs = Xs
        # self._y = y
        # self._y_unique = torch.unique(self._y) if y_unique is None else y_unique
        # self.ecs = [torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
        #             for clazz in self._y_unique]
        # self.n_views = self._Xs.shape[0]
        # self.n_samples = self._y.shape[0]
        # self.ori_dims = [X.shape[1] for X in self._Xs]
        # self._Xs = self.kernels.fit_transform(self._Xs)
        # self.dims = [X.shape[1] for X in self._Xs]
        self._post_prepare_()

    def __call__(self, *args, **kwargs):
        return self.O
