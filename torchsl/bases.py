

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


class Fittable:
    def __init__(self):
        self.is_fit: bool = False


class BaseAlgo(ResourcesPreparer, Fittable):
    def __init__(self, **kwargs):
        super(BaseAlgo, self).__init__()
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
