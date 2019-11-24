from .bases import BaseSLObjective
import torch


# ------------------------------------
# LDA
# ------------------------------------
class LDAIntraScatter(BaseSLObjective):

    def __init__(self):
        super(LDAIntraScatter, self).__init__(predicate='minimize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        D = torch.eye(self.n_samples)
        return self._X.t() @ (D - W) @ self._X


class LDAInterScatter(BaseSLObjective):

    def __init__(self):
        super(LDAInterScatter, self).__init__(predicate='maximize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * len(self._X))
        B = torch.ones(self.n_samples, self.n_samples) / self.n_samples
        return self._X.t() @ (W - B) @ self._X
