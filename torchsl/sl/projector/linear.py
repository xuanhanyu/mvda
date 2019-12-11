from torchsl.utils.typing import *
from torchsl.grad.constraints import stiefel_restore
import torch


class LinearProjector(torch.nn.Module):

    def __init__(self, ori_dim, projection_dim):
        super(LinearProjector, self).__init__()
        self.w = torch.nn.Parameter(stiefel_restore(torch.eye(ori_dim)[:, :projection_dim]), requires_grad=True)
        # nn.init.normal_(self.w)

    def forward(self, X: Tensor) -> Tensor:
        Y = X @ self.w
        return Y
