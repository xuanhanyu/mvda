from sklearn.metrics import pairwise
from .tensorutils import TensorUser, pre_vectorize, post_tensorize
from .typing import *
import torch


# ------------------------------------
# Template for Kernels
# ------------------------------------
class MvKernelizer(TensorUser):

    def __init__(self, kernels: Union[Callable, String, Sequence[Callable], Sequence[String]]):
        self.kernels: Sequence[Callable] = kernels
        self._Ls: Optional[Sequence[Optional[Tensorizable]]] = None
        self.n_views: Optional[Integer] = None
        self.statuses: Sequence[bool] = []
        self.is_fit: bool = False

    def _infer_kernels_funcs(self, Xs: Union[Tensor, Sequence[Tensor]]) -> None:
        self.n_views = len(Xs)
        self.statuses = [True for _ in range(self.n_views)]
        if not isinstance(self.kernels, list):
            self.kernels = [self.kernels for _ in range(self.n_views)]
        for ki in range(self.n_views):
            kernel = self.kernels[ki]
            if kernel is None:
                self.kernels[ki] = lambda x, l: x
                self.statuses[ki] = False
            if isinstance(kernel, str):
                kernel = kernel.lower()
                if kernel in ['lin', 'linear']:
                    self.kernels[ki] = pairwise.linear_kernel
                elif kernel in ['poly', 'polynomial']:
                    self.kernels[ki] = pairwise.polynomial_kernel
                elif kernel in ['rbf', 'gaussian']:
                    self.kernels[ki] = pairwise.rbf_kernel
                elif kernel in ['sigmoid']:
                    self.kernels[ki] = pairwise.sigmoid_kernel
                elif kernel in ['lap', 'laplacian']:
                    self.kernels[ki] = pairwise.laplacian_kernel
                elif kernel in ['chi2']:
                    self.kernels[ki] = pairwise.chi2_kernel
                elif kernel in ['achi2']:
                    self.kernels[ki] = pairwise.additive_chi2_kernel
                elif kernel in vars(pairwise):
                    self.kernels[ki] = vars(pairwise)[kernel]
                elif kernel in ['none']:
                    # None
                    self.kernels[ki] = lambda x, l: x
                    self.statuses[ki] = False
                elif kernel in ['precomputed']:
                    # Precomputed
                    self.kernels[ki] = lambda x, l: x
                else:
                    raise AttributeError('Undefined kernel type \"{}\"'.format(kernel))
            elif callable(kernel):
                pass

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
