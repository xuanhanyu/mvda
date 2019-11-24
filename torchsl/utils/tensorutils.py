from .typing import *
import numpy as np
import torch


# ------------------------------------
# Hidden function
# ------------------------------------
def _tensorize(mat: Any,
               dtype: Optional[Dtype] = None,
               device: Optional[Device] = None) -> Union[Tensor, Any]:
    if mat is None:
        return mat
    elif torch.is_tensor(mat):
        pass
    elif isinstance(mat, np.ndarray):
        mat = torch.from_numpy(mat)
    elif hasattr(mat, '__iter__'):
        return torch.stack([_tensorize(_) for _ in mat])
    else:
        mat = torch.tensor(mat)

    if dtype is not None:
        mat = mat.to(dtype=dtype)
    if device is not None:
        mat = mat.to(device=device)
    return mat


def _vectorize(mat: Any):
    if torch.is_tensor(mat) or isinstance(mat, np.ndarray):
        return mat
    return np.array(mat)


# ------------------------------------
# Decorators
# ------------------------------------
def pre_process(positionals: Union[Integer, Sequence[Integer]] = (),
                keywords: Union[String, Sequence[String]] = (),
                transform: Callable = lambda arg: arg):
    if not hasattr(positionals, '__iter__'):
        positionals = [positionals]
    if not hasattr(keywords, '__iter__') or isinstance(keywords, str):
        keywords = [keywords]

    def decorator(func):
        def wrapper(*args, **kwargs):
            args = list(args)
            for positional in positionals:
                args[positional] = transform(args[positional])
            for keyword in keywords:
                if keyword in kwargs:
                    kwargs[keyword] = transform(kwargs[keyword])
            return func(*args, **kwargs)

        return wrapper

    return decorator


def post_process(positionals: Union[Integer, Sequence[Integer]] = (),
                 transform: Callable = lambda arg: arg):
    if not hasattr(positionals, '__iter__'):
        positionals = [positionals]

    def decorator(func):
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if len(positionals) > 0 and isinstance(ret, tuple):
                ret = list(ret)
                for positional in positionals:
                    ret[positional] = transform(ret[positional])
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
    pass


def post_numpify(positionals: Union[Integer, Sequence[Integer]] = (),
                 dtype: Optional[Dtype] = None):
    pass


def pre_tensorize(positionals: Union[Integer, Sequence[Integer]] = (),
                  keywords: Union[String, Sequence[String]] = (),
                  dtype: Optional[Dtype] = None,
                  device: Optional[Device] = None):
    return pre_process(positionals, keywords, transform=lambda arg: _tensorize(arg, dtype=dtype, device=device))


def post_tensorize(positionals: Union[Integer, Sequence[Integer]] = (),
                   dtype: Optional[Dtype] = None,
                   device: Optional[Device] = None):
    return post_process(positionals, transform=lambda arg: _tensorize(arg, dtype=dtype, device=device))


def pre_vectorize(positionals: Union[Integer, Sequence[Integer]] = (),
                  keywords: Union[String, Sequence[String]] = ()):
    return pre_process(positionals, keywords, transform=lambda arg: _vectorize(arg))


def post_vectorize(positionals: Union[Integer, Sequence[Integer]] = ()):
    return post_process(positionals, transform=lambda arg: _vectorize(arg))


# ------------------------------------
# Classes
# ------------------------------------
class TensorUser:
    def _tensorize_(self, mat: Any,
                    dtype: Optional[Dtype] = None,
                    device: Optional[Device] = None) -> Tensor:
        return _tensorize(mat, dtype=dtype, device=device)

    def _vectorize_(self, mat: Any):
        return _vectorize(mat)
