from epsolver import EPSolver
import torch


# ------------------------------------
# Abstracts
# ------------------------------------
class Preparable:
    def __init__(self, **kwargs):
        self.keys = []
        self.dependents = []
        self.is_prepared = False

    def add_dependents_(self, *dependents):
        if isinstance(dependents, list):
            self.dependents.extend(dependents)
        else:
            self.dependents.extend([*dependents])

    def prepare_from_(self, other):
        for key in self.keys:
            if hasattr(other, key):
                setattr(self, key, getattr(other, key))
        self.is_prepared = True
        for dependent in self.dependents:
            dependent.prepare_from_(self)

    def prepare_(self, *args, **kwargs):
        self.is_prepared = True
        for dependent in self.dependents:
            dependent.prepare_from_(self)

    def check_prepared(self):
        valid = self.is_prepared
        for dependent in self.dependents:
            valid &= dependent.is_prepared
        return valid


class AlgoBase(Preparable):
    def __init__(self, **kwargs):
        super(AlgoBase, self).__init__()
        self.dependencies = []
        self.is_fit = False

    def add_dependencies_(self, *dependencies):
        if isinstance(dependencies, list):
            self.dependencies.extend(dependencies)
        else:
            self.dependencies.extend([*dependencies])

    def __fit__(self):
        self.is_fit = True
        for dependency in self.dependencies:
            dependency.__fit__()


# ------------------------------------
# Template for Algorithms
# ------------------------------------
class MvDAlgoBase(AlgoBase):
    name = 'Base Multi-view Discriminant Analysis Algorithm'

    def __init__(self, n_components='auto', ep='eig', reg='auto', *args, **kwargs):
        super(MvDAlgoBase, self).__init__()
        self.keys = ['_Xs', '_y', 'y_unique', 'ecs', 'n_views', 'n_samples', 'dims']
        self.n_components = n_components
        self.ep_solver = ep if isinstance(ep, EPSolver) else EPSolver(ep, reg)
        self.reg = reg

        # training buffers
        self.Sw = self.Sb = None
        self.eig_vecs = None
        self.ws = None

    def __fit__(self):
        super().__fit__()
        self.calculate_objectives()
        self.eig_vecs = self.ep_solver.solve(self.Sw, self.Sb)
        if self.n_components == 'auto':
            self.n_components = min(self.ep_solver.meaningful, min(self.dims))
        self.ws = [self.eig_vecs[sum(self.dims[:i]):sum(self.dims[:i + 1]), :] for i in range(len(self.dims))]

    def fit(self, Xs, y, y_unique=None):
        self.prepare_(Xs, y, y_unique)
        self.__fit__()

    def fit_like(self, other):
        self.prepare_from_(other)
        self.__fit__()

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
        Ys = [(self.ws[_][:, :self.n_components].t() @ Xs[_].t()).t() for _ in range(len(Xs))]
        return Ys

    def calculate_objectives(self):
        self.Sw = self.__Sw__()
        self.Sb = self.__Sb__()

    def __Sw__(self):
        raise NotImplementedError('Implement this base class first.')

    def __Sb__(self):
        raise NotImplementedError('Implement this base class first.')

    @property
    def class_vectors(self):
        return self.ecs

    @property
    def projections(self):
        return self.ws

    @property
    def W(self):
        return torch.cat(self.ws, dim=0)

    def prepare_(self, Xs, y, y_unique=None):
        if self.is_prepared:
            return
        self._Xs = Xs
        self._y = y
        self.y_unique = torch.unique(torch.tensor(y)) if y_unique is None else y_unique
        self.ecs = [torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                    for clazz in self.y_unique]
        self.n_views = len(self._Xs)
        self.n_samples = len(self._y)
        self.dims = [X.shape[1] for X in self._Xs]
        super(MvDAlgoBase, self).prepare_()

    def __str__(self):
        return self.name


# ------------------------------------
# Template for Objectives
# ------------------------------------
class MvDObjectiveBase(AlgoBase):
    name = 'Base Multi-view Discriminant Analysis Objective'

    def __init__(self, *args, **kwargs):
        super(MvDObjectiveBase, self).__init__()
        self.keys = ['_Xs', '_y', 'y_unique', 'ecs', 'n_views', 'n_samples', 'dims']
        self.O = None

    def __fit__(self):
        self.O = self.__O__()
        super().__fit__()

    def fit(self, Xs, y, y_unique=None):
        self.prepare_(Xs, y, y_unique)
        self.__fit__()

    def fit_like(self, other):
        self.prepare_from_(other)
        self.__fit__()

    def target(self):
        return self.O

    def __O__(self):
        raise NotImplementedError('Implement this base class first.')

    def prepare_(self, Xs, y, y_unique=None):
        if self.is_prepared:
            return
        self._Xs = Xs
        self._y = y
        self.y_unique = torch.unique(torch.tensor(y)) if y_unique is None else y_unique
        self.ecs = [torch.tensor([1 if _ == clazz else 0 for _ in self._y], dtype=torch.float)
                    for clazz in self.y_unique]
        self.n_views = len(self._Xs)
        self.n_samples = len(self._y)
        self.dims = [X.shape[1] for X in self._Xs]
        super(MvDObjectiveBase, self).prepare_()

    def __call__(self, *args, **kwargs):
        return self.O
