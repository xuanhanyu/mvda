from typing import *
from numpy import ndarray
import torch
import builtins


Number = Union[builtins.int, builtins.float, builtins.bool]
Integer = builtins.int
Float = builtins.float
String = builtins.str
Boolean = Union[builtins.bool, builtins.int, builtins.float, Any]

NumpyArray = ndarray
Tensor = torch.Tensor
Tensorizable = Union[torch.Tensor, NumpyArray]
Dtype = torch.dtype
Device = torch.device
