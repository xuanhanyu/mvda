from ..utils.typing import *
from scipy.linalg import fractional_matrix_power
import torch


class StiefelManifoldConstraint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp: Tensor) -> Tensor:
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_out: Optional[Tensor]) -> Optional[Tensor]:
        grad_in = None
        if ctx.needs_input_grad[0]:
            inp, = ctx.saved_tensors
            grad_in = grad_out.clone()
            grad_in = grad_in - inp @ grad_in.T @ inp
        return grad_in

    @staticmethod
    def restore(inp: Tensor) -> Tensor:
        data = inp.detach().cpu()
        rest = torch.from_numpy(fractional_matrix_power(data.t() @ data, -.5))
        return data @ rest


stiefel_constraint = StiefelManifoldConstraint.apply
stiefel_restore = StiefelManifoldConstraint.restore
