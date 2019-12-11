from .bases import GradientBasedSLAlgo
from .objectives import pcLDAObjective
from .projector import LinearProjector
from ..utils.typing import *
from ..grad.constraints import StiefelManifoldConstraint
import torch.nn as nn
import torch


class pcLDA(GradientBasedSLAlgo):

    def __init__(self, projector=LinearProjector(3, 2), q=1, beta=1):
        super(pcLDA, self).__init__(projector=projector)
        self.criterion = pcLDAObjective(projector=self.projector, q=q, beta=beta)
        self.constraint = StiefelManifoldConstraint.apply

    def forward(self,
                X: Tensor,
                y: Tensor,
                y_unique: Optional[Tensor] = None) -> Tensor:
        loss = self.criterion(X, y, y_unique)
        return loss
