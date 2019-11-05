from .utils import EPSolver
from .utils import MvKernels
from .utils import TensorUser
from abc import ABC, abstractmethod
import torch


# ------------------------------------
# Abstracts
# ------------------------------------
class Preparable:
    def __init__(self, **kwargs):
        self.keys = set()
        self.dependents = set()
        self.is_prepared = False

    def auto_chain(self):
        for val in vars(self).values():
            if isinstance(val, Preparable):
                self.add_dependents(val)

    def add_dependents(self, *dependents):
        if isinstance(dependents, list) or isinstance(dependents, set):
            self.dependents.update({dependents})
        else:
            self.dependents.update({*dependents})

    def _prepare_from_(self, other):
        for key in self.keys:
            if hasattr(other, key):
                setattr(self, key, getattr(other, key))
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


class AlgoBase(Preparable):
    def __init__(self, **kwargs):
        super(AlgoBase, self).__init__()
        self.dependencies = set()
        self.is_fit = False

    def auto_chain(self):
        for val in vars(self).values():
            if isinstance(val, AlgoBase):
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
KEYS = {'_Xs', '_y', '_y_unique', 'ecs', 'n_views', 'n_samples', 'ori_dims', 'dims', 'kernels'}


class MvDAlgoBase(AlgoBase, TensorUser, ABC):

    def __init__(self, n_components='auto', ep='eig', reg='auto', kernels=None, *args, **kwargs):
        super(MvDAlgoBase, self).__init__()
        self.keys = KEYS
        self.n_components = n_components
        self.ep_solver = ep if isinstance(ep, EPSolver) else EPSolver(ep, reg)
        self.reg = reg
        self.kernels = MvKernels(kernels)

        # training buffers
        self.Sw = self.Sb = None
        self.eig_vecs = None
        self.ws = None

    def _fit_(self):
        super()._fit_()
        self.calculate_objectives()
        self.eig_vecs = self.ep_solver.solve(self.Sw, self.Sb)
        if self.n_components == 'auto':
            self.n_components = min(self.ep_solver.meaningful, min(self.dims))
        elif self.n_components == 'same':
            assert int(sum(self.ori_dims) / self.n_views) == self.ori_dims[0]
            self.n_components = self.ori_dims[0]
        self.ws = [self.eig_vecs[sum(self.dims[:i]):sum(self.dims[:i + 1]), :] for i in range(len(self.dims))]

    def fit(self, Xs, y, y_unique=None):
        self._prepare_(Xs, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other):
        self._prepare_from_(other)
        self._fit_()
        return self

    def fit_transform(self, Xs, y, y_unique=None):
        self.fit(Xs, y, y_unique)
        return self.transform(Xs)

    def fit_transform_like(self, other):
        assert hasattr(other, '_Xs')
        _Xs = getattr(other, '_Xs')
        self.fit_like(other)
        return self.transform(_Xs)

    def transform(self, Xs):
        assert self.is_fit
        Xs = self.kernels.transform(Xs)
        Ys = [(self.ws[_][:, :self.n_components].t() @ Xs[_].t()).t() for _ in range(len(Xs))]
        return Ys

    def calculate_objectives(self):
        self.Sw = self._Sw_()
        self.Sb = self._Sb_()

    @abstractmethod
    def _Sw_(self):
        pass

    @abstractmethod
    def _Sb_(self):
        pass

    @property
    def predicates(self):
        predicates = {'maximize': [], 'minimize': []}
        for val in vars(self).values():
            if isinstance(val, MvDObjectiveBase):
                if val.predicate.lower().startswith('max'):
                    predicates['maximize'].append(str(val))
                elif val.predicate.lower().startswith('min'):
                    predicates['minimize'].append(str(val))
        return predicates

    @property
    def class_vectors(self):
        return self.ecs

    @property
    def projections(self):
        return self.ws

    @property
    def W(self):
        return torch.cat(self.ws, dim=0)

    def _prepare_(self, Xs, y, y_unique=None):
        if self.is_prepared:
            return
        self._Xs = [self._tensorize_(X) for X in Xs]
        self._y = self._tensorize_(y).long()
        self._y_unique = torch.unique(self._y) if y_unique is None else self._tensorize_(y_unique).long()
        self.ecs = [torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                    for clazz in self._y_unique]
        self.n_views = len(self._Xs)
        self.n_samples = len(self._y)
        self.ori_dims = [X.shape[1] for X in self._Xs]
        self._Xs = self.kernels.fit_transform(self._Xs)
        self.dims = [X.shape[1] for X in self._Xs]
        super()._prepare_()


# ------------------------------------
# Template for Objectives
# ------------------------------------
class MvDObjectiveBase(AlgoBase, TensorUser, ABC):

    def __init__(self, predicate='maximize', kernels=None, *args, **kwargs):
        super(MvDObjectiveBase, self).__init__()
        self.keys = KEYS
        self.O = None
        self.predicate = predicate
        self.kernels = MvKernels(kernels)

    def _fit_(self):
        self.O = self._O_()
        super()._fit_()

    def fit(self, Xs, y, y_unique=None):
        self._prepare_(Xs, y, y_unique)
        self._fit_()
        return self

    def fit_like(self, other):
        self._prepare_from_(other)
        self._fit_()
        return self

    def target(self):
        return self.O

    @abstractmethod
    def _O_(self):
        pass

    def _prepare_(self, Xs, y, y_unique=None):
        if self.is_prepared:
            return
        self._Xs = [self._tensorize_(X) for X in Xs]
        self._y = self._tensorize_(y).long()
        self._y_unique = torch.unique(self._y) if y_unique is None else self._tensorize_(y_unique).long()
        self.ecs = [torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                    for clazz in self._y_unique]
        self.n_views = len(self._Xs)
        self.n_samples = len(self._y)
        self.ori_dims = [X.shape[1] for X in self._Xs]
        self._Xs = self.kernels.fit_transform(self._Xs)
        self.dims = [X.shape[1] for X in self._Xs]
        super()._prepare_()

    def __call__(self, *args, **kwargs):
        return self.O
