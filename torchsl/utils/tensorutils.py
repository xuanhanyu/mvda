from .typing import *
import numpy as np
import torch


# ------------------------------------
# Hidden function
# ------------------------------------
def _tensor_type(mat: Any):
    if mat is None:
        pass
    elif torch.is_tensor(mat):
        return torch.Tensor
    elif isinstance(mat, np.ndarray):
        return np.ndarray
    elif hasattr(mat, '__iter__'):
        try:
            return _tensor_type(mat[0])
        except IndexError:
            pass
    return None


def _tensorize(mat: Any,
               dtype: Optional[Dtype] = None,
               device: Optional[Device] = None,
               requires_grad: Optional[Boolean] = None) -> Union[Tensor, Any]:
    if mat is None:
        pass
    elif torch.is_tensor(mat):
        mat = mat.to(dtype=dtype, device=device)
    elif isinstance(mat, np.ndarray):
        mat = torch.from_numpy(mat).to(dtype=dtype, device=device)
    elif hasattr(mat, '__iter__'):
        mat = torch.stack([_tensorize(_) for _ in mat]).to(dtype=dtype, device=device)
    else:
        mat = torch.tensor(mat, dtype=dtype, device=device)
    if requires_grad is not None and mat is not None and mat.dtype in [torch.float, torch.double]:
        mat.requires_grad = requires_grad
    return mat


def _numpify(mat: Any,
             dtype: Optional[Dtype] = None):
    if mat is None:
        pass
    elif isinstance(mat, np.ndarray):
        mat = mat.astype(dtype)
    if torch.is_tensor(mat):
        return mat.numpy().astype(dtype)
    elif hasattr(mat, '__iter__'):
        return np.array([_numpify(_) for _ in mat], dtype=dtype)
    return mat


def _vectorize(mat: Any):
    if mat is None:
        pass
    elif torch.is_tensor(mat) or isinstance(mat, np.ndarray):
        return mat
    elif hasattr(mat, '__iter__'):
        element_type = _tensor_type(mat)
        if element_type is np.ndarray:
            return np.array(mat)
        elif element_type is torch.Tensor:
            return _tensorize(mat)
    return mat


def _listify(mat: Tensorizable):
    if mat is None:
        pass
    if torch.is_tensor(mat) or isinstance(mat, np.ndarray):
        return mat.tolist()
    elif hasattr(mat, '__iter__'):
        return [_listify(_) for _ in mat]
    return mat


# ------------------------------------
# Decorators
# ------------------------------------
# class SelfArg:
#     def __init__(self, retriever):
#         self.retriever = retriever
#
#     def __call__(self, obj):
#         return self.retriever(obj)
#
#     @staticmethod
#     def unpack(obj, args: Dict):
#         for arg in args:
#             retr = args[arg]
#             if isinstance(retr, SelfArg):
#                 args[arg] = retr(obj)


def pre_process(positionals: Union[Integer, Sequence[Integer]] = (),
                keywords: Union[String, Sequence[String]] = (),
                transform: Callable = lambda arg: arg,
                transform_args: Dict = {}):
    if not hasattr(positionals, '__iter__'):
        positionals = [positionals]
    if not hasattr(keywords, '__iter__') or isinstance(keywords, str):
        keywords = [keywords]

    def decorator(func):
        def wrapper(*args, **kwargs):
            args = list(args)
            for positional in positionals:
                if len(args) > positional:
                    args[positional] = transform(args[positional], **transform_args)
            for keyword in keywords:
                if keyword in kwargs:
                    kwargs[keyword] = transform(kwargs[keyword], **transform_args)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def post_process(positionals: Union[Integer, Sequence[Integer]] = (),
                 transform: Callable = lambda arg: arg,
                 transform_args: Dict = {}):
    if not hasattr(positionals, '__iter__'):
        positionals = [positionals]

    def decorator(func):
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if len(positionals) > 0 and isinstance(ret, tuple):
                ret = list(ret)
                for positional in positionals:
                    if len(ret) > positional:
                        ret[positional] = transform(ret[positional], **transform_args)
                return tuple(ret)
            return transform(ret)

        return wrapper

    return decorator


def process(pre_positionals: Union[Integer, Sequence[Integer]] = (),
            pre_keywords: Union[String, Sequence[String]] = (),
            post_positionals: Union[Integer, Sequence[Integer]] = (),
            pre_transform: Callable = lambda arg: arg,
            post_transform: Callable = lambda arg: arg):
    if not hasattr(pre_positionals, '__iter__'):
        pre_positionals = [pre_positionals]
    if not hasattr(pre_keywords, '__iter__') or isinstance(pre_keywords, str):
        pre_keywords = [pre_keywords]
    if not hasattr(post_positionals, '__iter__'):
        post_positionals = [post_positionals]

    def decorator(func):
        def wrapper(*args, **kwargs):
            # pre-process
            args = list(args)
            for positional in pre_positionals:
                args[positional] = pre_transform(args[positional])
            for keyword in pre_keywords:
                if keyword in kwargs:
                    kwargs[keyword] = pre_transform(kwargs[keyword])
            ret = func(*args, **kwargs)
            # post-process
            if len(post_positionals) > 0 and isinstance(ret, tuple):
                ret = list(ret)
                for positional in post_positionals:
                    ret[positional] = post_transform(ret[positional])
                return tuple(ret)
            return post_transform(ret)

        return wrapper

    return decorator


def pre_numpify(positionals: Union[Integer, Sequence[Integer]] = (),
                keywords: Union[String, Sequence[String]] = (),
                dtype: Optional[Dtype] = None):
    return pre_process(positionals, keywords, transform=lambda arg: _numpify(arg, dtype=dtype))


def post_numpify(positionals: Union[Integer, Sequence[Integer]] = (),
                 dtype: Optional[Dtype] = None):
    return post_process(positionals, transform=lambda arg: _numpify(arg, dtype=dtype))


def pre_tensorize(positionals: Union[Integer, Sequence[Integer]] = (),
                  keywords: Union[String, Sequence[String]] = (),
                  dtype: Optional[Dtype] = None,
                  device: Optional[Device] = None,
                  requires_grad: Optional[Boolean] = None):
    return pre_process(positionals, keywords,
                       transform=lambda arg: _tensorize(arg, dtype=dtype, device=device, requires_grad=requires_grad))


def post_tensorize(positionals: Union[Integer, Sequence[Integer]] = (),
                   dtype: Optional[Dtype] = None,
                   device: Optional[Device] = None):
    return post_process(positionals, transform=lambda arg: _tensorize(arg, dtype=dtype, device=device))


def pre_vectorize(positionals: Union[Integer, Sequence[Integer]] = (),
                  keywords: Union[String, Sequence[String]] = ()):
    return pre_process(positionals, keywords, transform=lambda arg: _vectorize(arg))


def post_vectorize(positionals: Union[Integer, Sequence[Integer]] = ()):
    return post_process(positionals, transform=lambda arg: _vectorize(arg))


def pre_listify(positionals: Union[Integer, Sequence[Integer]] = (),
                keywords: Union[String, Sequence[String]] = ()):
    return pre_process(positionals, keywords, transform=lambda arg: _listify(arg))


def post_listify(positionals: Union[Integer, Sequence[Integer]] = ()):
    return post_process(positionals, transform=lambda arg: _listify(arg))


# ------------------------------------
# Classes
# ------------------------------------
class TensorUser:
    def __init__(self, reg: Union[Number, String] = 'auto',
                 device: Optional[Device] = None,
                 requires_grad: Boolean = None):
        self.reg: Union[Number, String] = reg
        self.device: Optional[Device] = device
        self.requires_grad: bool = requires_grad

    def _tensorize(self, mat: Any,
                   dtype: Optional[Dtype] = None) -> Tensor:
        return _tensorize(mat, dtype=dtype, device=self.device, requires_grad=self.requires_grad)

    def _numpify(self, mat: Any,
                 dtype: Optional[Dtype] = None) -> NumpyArray:
        return _numpify(mat, dtype=dtype)

    def _vectorize(self, mat: Any):
        return _vectorize(mat)

    def _listify(self, mat: Tensorizable):
        return _listify(mat)

    @pre_vectorize(positionals=1)
    def _regularize(self, mat: Any):
        if torch.is_tensor(mat):
            if self.reg == 'auto':
                diag = torch.trace(mat)
                if diag == 0:
                    diag = 1e-5
                I_reg = torch.eye(mat.shape[0]) * diag * 1e-4
            else:
                I_reg = torch.eye(mat.shape[0]) * self.reg
        else:
            if self.reg == 'auto':
                diag = np.trace(mat)
                if diag == 0:
                    diag = 1e-5
                I_reg = np.eye(mat.shape[0]) * diag * 1e-4
            else:
                I_reg = np.eye(mat.shape[0]) * self.reg
        return mat + I_reg
