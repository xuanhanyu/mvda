import numpy as np
import torch


def pre_process(positionals=(), keywords=(), transform=lambda arg: arg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            args = list(args)
            for positional in positionals:
                args[positional] = transform(args[positional])
            for keyword in keywords:
                kwargs[keyword] = transform(kwargs[keyword])
            return func(*args, **kwargs)
        return wrapper
    return decorator


def post_process(positionals=(), transform=lambda arg: arg):
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


def tensorize(arg, dtype=None, device=None):
    if torch.is_tensor(arg):
        ret = arg
    elif isinstance(arg, np.ndarray):
        ret = torch.from_numpy(arg)
    else:
        ret = torch.tensor(arg)
    if dtype is not None:
        ret = ret.to(dtype=dtype)
    if device is not None:
        ret = ret.to(device=device)
    return ret


def pre_tensorize(positionals=(), keywords=(), dtype=None, device=None):
    return pre_process(positionals, keywords, transform=lambda arg: tensorize(arg, dtype=dtype, device=device))


def post_tensorize(positionals=(), dtype=None, device=None):
    return post_process(positionals, transform=lambda arg: tensorize(arg, dtype=dtype, device=device))


class Dummy:
    @pre_tensorize(positionals=[1])
    @post_tensorize(dtype=torch.float)
    def foo(self, x, *args, dim=0, **kwargs):
        return x


if __name__ == '__main__':
    dm = Dummy()
    ret = dm.foo(np.random.rand(2, 2), 1, 2, dim=1, bar=2)
    print(ret)
