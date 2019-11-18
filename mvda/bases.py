from .utils import EPSolver, EPAlgo, EPImplementation
from .utils import MvKernelizer
from .utils import TensorUser, pre_tensorize
from abc import ABC, abstractmethod
from .utils.typing import *
import torch


# ------------------------------------
# Abstracts
# ------------------------------------
class ResourcesPreparer:
    def __init__(self, **kwargs):
        self.sharable_resources = set()
        self.dependents = set()
        self.is_prepared = False

    def set_sharable_resources(self, sharable_resources):
        self.sharable_resources = sharable_resources

    def auto_chain(self):
        for val in vars(self).values():
            if isinstance(val, ResourcesPreparer):
                self.add_dependents(val)

    def add_dependents(self, *dependents):
        if isinstance(dependents, list) or isinstance(dependents, set):
            self.dependents.update({dependents})
        else:
            self.dependents.update({*dependents})

    def _prepare_from_(self, other):
        for resource in self.sharable_resources:
            if hasattr(other, resource):
                setattr(self, resource, getattr(other, resource))
        self.is_prepared = True
        self.auto_chain()
        for dependent in self.dependents:
            dependent._prepare_from_(self)

    def _prepare_(self, *args, **kwargs):
        self.is_prepared = True
        self.auto_chain()
        for dependent in self.dependents:
            dependent._prepare_from_(self)

    def check_prepared(self):
        valid = self.is_prepared
        for dependent in self.dependents:
            valid &= dependent.is_prepared
        return valid


class BaseAlgo(ResourcesPreparer):
    def __init__(self, **kwargs):
        super(BaseAlgo, self).__init__()
        self.dependencies = set()
        self.is_fit = False

    def auto_chain(self):
        for val in vars(self).values():
            if isinstance(val, BaseAlgo):
                self.add_dependents(val)
                self.add_dependencies(val)

    def add_dependencies(self, *dependencies):
        if isinstance(dependencies, list) or isinstance(dependencies, set):
            self.dependencies.update({dependencies})
        else:
            self.dependencies.update({*dependencies})

    def _fit_(self):
        self.is_fit = True
        for dependency in self.dependencies:
            dependency._fit_()


# ------------------------------------
# Template for Algorithms
# ------------------------------------
SHARABLE_RESOURCES = {'_Xs', '_y', '_y_unique', 'ecs', 'n_views', 'n_samples', 'ori_dims', 'dims', 'kernels'}


class BaseMvDAlgo(BaseAlgo, TensorUser, ABC):

    def __init__(self,
                 n_components: Union[Integer, String] = 'auto',
                 ep_algo: Union[EPAlgo, String] = 'eig',
                 ep_implementation: Union[EPImplementation, String] = 'pytorch',
                 reg: Union[Number, String] = 'auto',
                 kernels: Optional[Union[Callable, String, Iterable[Callable], Iterable[String]]] = None,
                 *args, **kwargs):
        super(BaseMvDAlgo, self).__init__()
        self.set_sharable_resources(SHARABLE_RESOURCES)
        self.n_components: Integer = n_components
        self.ep_solver: EPSolver = ep_algo if isinstance(ep_algo, EPSolver) else EPSolver(algo=ep_algo,
                                                                                          implementation=ep_implementation,
                                                                                          reg=reg)
        self.reg: Union[Number, String] = reg
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

    @pre_tensorize(positionals=(1, 2), keywords='y_unique')
    def fit(self, Xs: Union[Tensor, Sequence[Tensor]],
            y: Union[Tensor, Sequence[Any]],
            y_unique: Optional[Union[Tensor, Sequence[Any]]] = None) -> 'BaseMvDAlgo':
        self._prepare_(Xs, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other: 'BaseAlgo') -> 'BaseMvDAlgo':
        assert other.is_fit, '{} is not fitted yet!'.format(other)
        self._prepare_from_(other)
        self._fit_()
        return self

    @pre_tensorize(positionals=(1, 2), keywords='y_unique')
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

    @pre_tensorize(positionals=1)
    def transform(self, Xs: Union[Tensor, Sequence[Tensor]]) -> Tensor:
        assert self.is_fit
        Xs = self.kernels.transform(Xs)
        Ys = torch.stack([Xs[_] @ self.ws[_][:, :self.n_components] for _ in range(Xs.shape[0])])
        return Ys

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
            if isinstance(val, BaseMvDObjective):
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
        if self.is_prepared:
            return
        self._Xs = Xs
        self._y = y
        self._y_unique = torch.unique(self._y) if y_unique is None else y_unique
        self.ecs = torch.stack([torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                                for clazz in self._y_unique])
        self.n_views = self._Xs.shape[0]
        self.n_samples = self._y.shape[0]
        self.ori_dims = [X.shape[1] for X in self._Xs]
        self._Xs = self.kernels.fit_transform(self._Xs)
        self.dims = [X.shape[1] for X in self._Xs]
        super()._prepare_()


# ------------------------------------
# Template for Objectives
# ------------------------------------
class BaseMvDObjective(BaseAlgo, TensorUser, ABC):

    def __init__(self,
                 predicate: String = 'maximize',
                 kernels: Optional[Union[Callable, String, Iterable[Callable], Iterable[String]]] = None,
                 *args, **kwargs):
        super(BaseMvDObjective, self).__init__()
        self.set_sharable_resources(SHARABLE_RESOURCES)
        self.O = None
        self.predicate = predicate
        self.kernels = MvKernelizer(kernels)

    def _fit_(self) -> None:
        self.O = self._O_()
        super()._fit_()

    @pre_tensorize(positionals=(1, 2), keywords='y_unique')
    def fit(self,
            Xs: Union[Tensor, Iterable[Tensor]],
            y: Union[Tensor, NumpyArray, Iterable],
            y_unique: Optional[Union[Tensor, NumpyArray, Iterable]] = None) -> 'BaseMvDObjective':
        self._prepare_(Xs, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other: 'BaseAlgo') -> 'BaseMvDObjective':
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
                  Xs: Tensor,
                  y: Tensor,
                  y_unique: Optional[Tensor] = None) -> None:
        if self.is_prepared:
            return
        self._Xs = Xs
        self._y = y
        self._y_unique = torch.unique(self._y) if y_unique is None else y_unique
        self.ecs = [torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                    for clazz in self._y_unique]
        self.n_views = self._Xs.shape[0]
        self.n_samples = self._y.shape[0]
        self.ori_dims = [X.shape[1] for X in self._Xs]
        self._Xs = self.kernels.fit_transform(self._Xs)
        self.dims = [X.shape[1] for X in self._Xs]
        super()._prepare_()

    def __call__(self, *args, **kwargs):
        return self.O
