from .bases import MvDAlgoBase
from .objectives import MvDAInterScatter, CommonComponent, DifferingClass


class MvCCDA(MvDAlgoBase):
    name = 'Multi-view Common Component Discriminant Analysis'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 *args, **kwargs):
        super(MvCCDA, self).__init__(n_components=n_components,
                                     ep=ep,
                                     reg=reg,
                                     kernels=kernels,
                                     *args, **kwargs)
        self.cco = CommonComponent()
        self.sbo = MvDAInterScatter()

    def _Sw_(self):
        return self.cco.target()

    def _Sb_(self):
        return self.sbo.target()


class MvDCCCDA(MvDAlgoBase):
    name = 'Multi-view Differing Class Common Component Discriminant Analysis'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 *args, **kwargs):
        super(MvDCCCDA, self).__init__(n_components=n_components,
                                       ep=ep,
                                       reg=reg,
                                       kernels=kernels,
                                       *args, **kwargs)
        self.cco = CommonComponent()
        self.dco = DifferingClass()

    def _Sw_(self):
        return self.cco.target()

    def _Sb_(self):
        return self.dco.target()
