from .bases import EOBasedMvSLAlgo
from .objectives import MvDAInterScatter, CommonComponent, DifferingClass
from .mvda import MvDA


class MvCCDA(MvDA):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 lambda_cc=0.1,
                 *args, **kwargs):
        super(MvCCDA, self).__init__(n_components=n_components,
                                     ep_algo=ep_algo,
                                     ep_implementation=ep_implementation,
                                     reg=reg,
                                     kernels=kernels,
                                     *args, **kwargs)
        self.cco = CommonComponent()
        self.lambda_cc = lambda_cc

    def calculate_objectives(self) -> None:
        self.Sw = self._Sw_() + self.lambda_cc * self.cco.target()
        self.Sb = self._Sb_()


class MvDCCCDA(EOBasedMvSLAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 *args, **kwargs):
        super(MvDCCCDA, self).__init__(n_components=n_components,
                                       ep_algo=ep_algo,
                                       ep_implementation=ep_implementation,
                                       reg=reg,
                                       kernels=kernels,
                                       *args, **kwargs)
        self.cco = CommonComponent()
        self.dco = DifferingClass()

    def _Sw_(self):
        return self.cco.target()

    def _Sb_(self):
        return self.dco.target()
