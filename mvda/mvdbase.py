from epsolver import EPSolver
import torch


class Preparable:
    def __init__(self, keys=()):
        self.keys = keys
        self.is_prepared = False

    def _prepare_from_(self, other):
        for key in self.keys:
            if hasattr(other, key):
                setattr(self, key, getattr(other, key))
        self._post_prepare_()

    def _prepare_(self, *args, **kwargs):
        self._post_prepare_()

    def _post_prepare_(self):
        pass


class MvDAlgoBase(Preparable):
    name = 'Base Multi-view Discriminant Analysis Algorithm'

    def __init__(self, n_components='auto', ep='eig', reg='auto', *args, **kwargs):
        super(MvDAlgoBase, self).__init__(keys=['_Xs', '_y', 'y_unique', 'ecs', 'n_views', 'n_samples', 'dims'])
        self.n_components = n_components
        self.ep_solver = ep if isinstance(ep, EPSolver) else EPSolver(ep, reg)
        self.reg = reg
        self.is_fit = False

        # training buffers
        self.Sw = self.Sb = None
        self.eig_vecs = None
        self.ws = None

    def __fit__(self):
        self.__SwSb__()
        self.eig_vecs = self.ep_solver.solve(self.Sw, self.Sb)
        if self.n_components == 'auto':
            self.n_components = self.ep_solver.meaningful
        self.ws = [self.eig_vecs[sum(self.dims[:i]):sum(self.dims[:i + 1]), :] for i in range(len(self.dims))]
        del self._Xs
        del self._y
        del self.ecs
        self.is_fit = True

    def fit(self, Xs, y, y_unique=None):
        self._prepare_(Xs, y, y_unique=y_unique)
        self.__fit__()

    def fit_transform(self, Xs, y, y_unique=None):
        self.fit(Xs, y, y_unique)
        return self.transform(Xs)

    def transform(self, Xs):
        assert self.is_fit
        Ys = [(self.ws[_][:, :self.n_components].t() @ Xs[_].t()).t() for _ in range(len(Xs))]
        return Ys

    def __SwSb__(self):
        self.Sw = self.__Sw__()
        self.Sb = self.__Sb__()

    def __Sw__(self):
        raise NotImplementedError('Implement this base class first.')

    def __Sb__(self):
        raise NotImplementedError('Implement this base class first.')

    @property
    def projections(self):
        return self.ws

    def _prepare_(self, Xs, y, y_unique=None):
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
        self._post_prepare_()

    def __str__(self):
        return self.name


class MvDObjectiveBase(Preparable):
    name = 'Base Multi-view Discriminant Analysis Objective'

    def __init__(self, *args, **kwargs):
        super(MvDObjectiveBase, self).__init__(keys=['_Xs', '_y', 'y_unique', 'ecs', 'n_views', 'n_samples', 'dims'])
        self.O = None

    def __fit__(self):
        self.O = self.__O__()
        del self._Xs
        del self._y
        del self.ecs

    def fit(self, Xs, y, y_unique=None):
        self._prepare_(Xs, y, y_unique)
        self.__fit__()

    def fit_like(self, other):
        self._prepare_from_(other)
        self.__fit__()

    def target(self):
        return self.O

    def __O__(self):
        raise NotImplementedError('Implement this base class first.')

    def _prepare_(self, Xs, y, y_unique=None):
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
        self._post_prepare_()
