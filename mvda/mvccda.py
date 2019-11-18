from .bases import BaseMvDAlgo
from .objectives import MvDAInterScatter, CommonComponent, DifferingClass


class MvCCDA(BaseMvDAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 *args, **kwargs):
        super(MvCCDA, self).__init__(n_components=n_components,
                                     ep_algo=ep_algo,
                                     ep_implementation=ep_implementation,
                                     reg=reg,
                                     kernels=kernels,
                                     *args, **kwargs)
        self.cco = CommonComponent()
        self.sbo = MvDAInterScatter()

    def _Sw_(self):
        return self.cco.target()

    def _Sb_(self):
        return self.sbo.target()


class MvDCCCDA(BaseMvDAlgo):

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
