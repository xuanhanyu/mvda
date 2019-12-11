from .utils.typing import *
from .utils.tensorutils import pre_tensorize


# ------------------------------------
# Abstracts
# ------------------------------------
class ResourcesPreparer:
    def __init__(self, sharable_resources=(), **kwargs):
        self.sharable_resources = set(sharable_resources)
        self.dependents = set()
        self.is_prepared = False
        self._should_reprepare = True

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
            dependent.__should_reprepare = False
            dependent._prepare_from_(self)
            dependent.__should_reprepare = True

    def _prepare_(self, *args, **kwargs):
        pass

    def _post_prepare_(self):
        self.is_prepared = True
        self.auto_chain()
        for dependent in self.dependents:
            dependent.__should_reprepare = False
            dependent._prepare_from_(self)
            dependent.__should_reprepare = True

    def check_prepared(self):
        valid = self.is_prepared
        for dependent in self.dependents:
            valid &= dependent.is_prepared
        return valid


class Fittable:
    def __init__(self):
        self.is_fit: bool = False


class BaseAlgo(ResourcesPreparer, Fittable):
    def __init__(self, sharable_resources=(), **kwargs):
        ResourcesPreparer.__init__(self, sharable_resources=sharable_resources)
        Fittable.__init__(self)
        self.dependencies = set()

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
# Metas
# ------------------------------------
class MetaEOBasedAlgo(type):

    def __init__(cls, name, bases, attr):
        for func in ['fit', 'fit_transform']:
            if func in attr:
                pretensorized_func = pre_tensorize(positionals=1, dtype=torch.float)(
                    pre_tensorize(positionals=(2, 3), keywords='y_unique', dtype=torch.long)(attr[func])
                )
                setattr(cls, func, pretensorized_func)
        for func in ['transform']:
            if func in attr:
                pretensorized_func = pre_tensorize(positionals=1, dtype=torch.float)(attr[func])
                setattr(cls, func, pretensorized_func)


class MetaGradientBasedAlgo(type):

    @staticmethod
    def prepare(cls, func):
        def wrapper(*args, **kwargs):
            if hasattr(cls, '_prepare_'):
                getattr(cls, '_prepare_')(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper

    def __init__(cls, name, bases, attr):
        for func in ['forward']:
            if func in attr:
                pretensorized_func = pre_tensorize(positionals=1, dtype=torch.float, requires_grad=None)(
                    pre_tensorize(positionals=(2, 3), keywords='y_unique', dtype=torch.long)(
                        MetaGradientBasedAlgo.prepare(cls, attr[func])
                    )
                )
                setattr(cls, func, pretensorized_func)
        for func in ['transform']:
            if func in attr:
                pretensorized_func = pre_tensorize(positionals=1, dtype=torch.float)(attr[func])
                setattr(cls, func, pretensorized_func)
