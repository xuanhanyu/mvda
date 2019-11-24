from sklearn.metrics import pairwise
from ..bases import Fittable
from .tensorutils import TensorUser, pre_vectorize, post_tensorize
from .typing import *
import torch


def _get_kernel_func(kernel: Union[Callable, String]) -> Tuple[Callable, bool]:
    use_kernel = True
    if kernel is None:
        kernel = lambda x, l: x
        use_kernel = False
    if isinstance(kernel, str):
        kernel = kernel.lower()
        if kernel in ['lin', 'linear']:
            kernel = pairwise.linear_kernel
        elif kernel in ['poly', 'polynomial']:
            kernel = pairwise.polynomial_kernel
        elif kernel in ['rbf', 'gaussian']:
            kernel = pairwise.rbf_kernel
        elif kernel in ['sigmoid']:
            kernel = pairwise.sigmoid_kernel
        elif kernel in ['lap', 'laplacian']:
            kernel = pairwise.laplacian_kernel
        elif kernel in ['chi2']:
            kernel = pairwise.chi2_kernel
        elif kernel in ['achi2']:
            kernel = pairwise.additive_chi2_kernel
        elif kernel in vars(pairwise):
            kernel = vars(pairwise)[kernel]
        elif kernel in ['none']:
            # None
            kernel = lambda x, l: x
            use_kernel = False
        elif kernel in ['precomputed']:
            # Precomputed
            kernel = lambda x, l: x
        else:
            raise AttributeError('Undefined kernel type \"{}\"'.format(kernel))
    elif callable(kernel):
        pass
    return kernel, use_kernel


# ------------------------------------
# Template for Kernel
# ------------------------------------
class Kernelizer(Fittable, TensorUser):

    def __init__(self, kernel: Union[Callable, String]):
        super(Kernelizer, self).__init__()
        self.kernel: Callable = kernel
        self._L: Optional[Tensorizable] = None
        self.status: bool = True
        self.is_fit: bool = False

    def _infer_kernel_func(self, X: Tensor) -> None:
        self.kernel, self.status = _get_kernel_func(self.kernel)

    @pre_vectorize(positionals=1)
    def fit(self, X: Tensorizable) -> 'Kernelizer':
        if not self.is_fit:
            self._infer_kernel_func(X)
            self.is_fit = True
        self._L = X.data.clone() if self.status else None
        return self

    @pre_vectorize(positionals=1)
    def fit_transform(self, X: Tensorizable) -> Tensor:
        self.fit(X)
        return self.transform(X)

    @pre_vectorize(positionals=1)
    @post_tensorize(dtype=torch.float)
    def transform(self, X: Tensorizable) -> Tensor:
        assert self.is_fit, "Not fitted yet!"
        return self.kernel(X, self._L)


class MvKernelizer(Fittable, TensorUser):

    def __init__(self, kernels: Union[Callable, String, Sequence[Callable], Sequence[String]]):
        super(MvKernelizer, self).__init__()
        self.kernels: Sequence[Callable] = kernels
        self._Ls: Optional[Sequence[Optional[Tensorizable]]] = None
        self.n_views: Optional[Integer] = None
        self.statuses: Sequence[bool] = []

    def _infer_kernels_funcs(self, Xs: Union[Tensor, Sequence[Tensor]]) -> None:
        self.n_views = len(Xs)
        self.statuses = [True for _ in range(self.n_views)]
        if not isinstance(self.kernels, list):
            self.kernels = [self.kernels for _ in range(self.n_views)]
        for ki in range(self.n_views):
            self.kernels[ki], self.statuses[ki] = _get_kernel_func(self.kernels[ki])

    @pre_vectorize(positionals=1)
    def fit(self, Xs: Union[Tensorizable, Sequence[Tensorizable]]) -> 'MvKernelizer':
        if not self.is_fit:
            self._infer_kernels_funcs(Xs)
            self.is_fit = True
        self._Ls = [Xs[_].data.clone() if self.statuses[_] else None for _ in range(self.n_views)]
        return self

    @pre_vectorize(positionals=1)
    def fit_transform(self, Xs: Union[Tensorizable, Sequence[Tensorizable]]) -> Tensor:
        self.fit(Xs)
        return self.transform(Xs)

    @pre_vectorize(positionals=1)
    @post_tensorize(dtype=torch.float)
    def transform(self, Xs: Union[Tensorizable, Sequence[Tensorizable]]) -> Union[Tensor, Any]:
        assert self.is_fit, 'Not fitted yet!'
        return [self.kernels[_](Xs[_], self._Ls[_]) for _ in range(self.n_views)]

    @property
    def all_activated(self) -> bool:
        ret = True
        for status in self.statuses:
            ret &= status
        return ret
