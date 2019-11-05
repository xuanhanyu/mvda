from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
    laplacian_kernel,
    chi2_kernel,
    additive_chi2_kernel)
from .torchutils import TensorUser
import torch


# ------------------------------------
# Template for Kernels
# ------------------------------------
class MvKernels(TensorUser):
    def __init__(self, kernels):
        self.kernels = kernels
        self._Ls = self.n_views = None
        self.activated = []
        self.is_fit = False

    def __infer_kernels__(self, Xs):
        self.n_views = len(Xs)
        self.activated = [True for _ in range(self.n_views)]
        if not isinstance(self.kernels, list):
            self.kernels = [self.kernels for _ in range(self.n_views)]
        for ki in range(self.n_views):
            kernel = self.kernels[ki]
            if kernel is None:
                self.kernels[ki] = lambda x, l: x
                self.activated[ki] = False
            if isinstance(kernel, str):
                kernel = kernel.lower()
                if kernel in ['lin', 'linear']:
                    self.kernels[ki] = linear_kernel
                elif kernel in ['poly', 'polynomial']:
                    self.kernels[ki] = polynomial_kernel
                elif kernel in ['rbf', 'gaussian']:
                    self.kernels[ki] = rbf_kernel
                elif kernel in ['sigmoid']:
                    self.kernels[ki] = sigmoid_kernel
                elif kernel in ['lap', 'laplacian']:
                    self.kernels[ki] = laplacian_kernel
                elif kernel in ['chi2']:
                    self.kernels[ki] = chi2_kernel
                elif kernel in ['achi2']:
                    self.kernels[ki] = additive_chi2_kernel
                elif kernel in ['none']:
                    # None
                    self.kernels[ki] = lambda x, l: x
                    self.activated[ki] = False
                elif kernel in ['precomputed']:
                    # Precomputed
                    self.kernels[ki] = lambda x, l: x
                else:
                    raise AttributeError('Undefined kernel type \"{}\"'.format(kernel))
            elif callable(kernel):
                pass

    def fit(self, Xs):
        if not self.is_fit:
            self.__infer_kernels__(Xs)
            self.is_fit = True
        self._Ls = [Xs[_].data.clone() if self.activated[_] else None for _ in range(self.n_views)]

    def fit_transform(self, Xs):
        self.fit(Xs)
        return self.transform(Xs)

    def transform(self, Xs):
        assert self.is_fit
        return [self._tensorize_(self.kernels[_](Xs[_], self._Ls[_])).float() for _ in range(self.n_views)]

    @property
    def all_activated(self):
        ret = True
        for status in self.activated:
            ret &= status
        return ret
